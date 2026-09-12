use std::collections::BTreeMap;

use anyhow::Result;
use candle_core::{D, Device, Tensor};
use uvr_backend_probe::{LstmDirection, Probe, Step, measure, rotary_tables};

fn main() -> Result<()> {
    let probe = Probe::load()?;
    let tensors: BTreeMap<String, Tensor> = probe
        .tensors
        .iter()
        .map(|(name, data)| {
            Ok((
                name.clone(),
                Tensor::from_vec(data.values.clone(), data.shape.as_slice(), &Device::Cpu)?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut rotary = BTreeMap::new();
    for case in &probe.cases {
        for step in &case.steps {
            if let Step::RoformerAttention {
                rotary_frequencies, ..
            } = step
            {
                let (cos, sin) = rotary_tables(
                    &probe.tensors[rotary_frequencies],
                    probe.tensors[&case.input].shape[1],
                );
                let tensor = |data: uvr_backend_probe::Data| {
                    Tensor::from_vec(data.values, data.shape.as_slice(), &Device::Cpu)
                };
                rotary.insert(case.name.clone(), (tensor(cos)?, tensor(sin)?));
            }
        }
    }
    let mut measurements = Vec::new();
    for case in &probe.cases {
        measurements.push(measure(&case.name, &probe.tensors[&case.expected], || {
            let mut x = tensors[&case.input].clone();
            for step in &case.steps {
                x = match step {
                    Step::Conv2d {
                        weight,
                        padding,
                        stride,
                        dilation,
                        groups,
                    } => x.conv2d(&tensors[weight], *padding, *stride, *dilation, *groups)?,
                    Step::BatchNorm {
                        weight,
                        bias,
                        mean,
                        variance,
                        epsilon,
                    } => {
                        let channels = tensors[weight].elem_count();
                        let channel = |name: &str| tensors[name].reshape((1, channels, 1, 1));
                        let centered = x.broadcast_sub(&channel(mean)?)?;
                        let scale = (channel(variance)? + f64::from(*epsilon))?.sqrt()?;
                        centered
                            .broadcast_div(&scale)?
                            .broadcast_mul(&channel(weight)?)?
                            .broadcast_add(&channel(bias)?)?
                    }
                    Step::LeakyRelu { slope } => {
                        (x.maximum(0_f32)? + (x.minimum(0_f32)? * f64::from(*slope))?)?
                    }
                    Step::Bilinear {
                        size,
                        align_corners,
                    } => x.upsample_bilinear2d(size[0], size[1], *align_corners)?,
                    Step::BiLstm { forward, reverse } => Tensor::cat(
                        &[
                            lstm_direction(&x, forward, &tensors, false)?,
                            lstm_direction(&x, reverse, &tensors, true)?,
                        ],
                        2,
                    )?,
                    Step::RoformerAttention { .. } => {
                        attention(&x, step, &tensors, &rotary[&case.name])?
                    }
                    Step::RoformerFeedForward {
                        norm,
                        input_weight,
                        input_bias,
                        output_weight,
                        output_bias,
                    } => {
                        let (batch, sequence, dim) = x.dims3()?;
                        let normalized =
                            rms_norm(&x, &tensors[norm])?.reshape((batch * sequence, dim))?;
                        let hidden = normalized
                            .matmul(&tensors[input_weight].t()?)?
                            .broadcast_add(&tensors[input_bias])?
                            .gelu_erf()?;
                        hidden
                            .matmul(&tensors[output_weight].t()?)?
                            .broadcast_add(&tensors[output_bias])?
                            .reshape((batch, sequence, dim))?
                    }
                };
            }
            Ok((x.dims().to_vec(), x.flatten_all()?.to_vec1::<f32>()?))
        })?);
    }
    probe.write(
        "candle-core 0.10.2 CPU, default features disabled",
        measurements,
    )
}

fn sigmoid(x: &Tensor) -> candle_core::Result<Tensor> {
    (x.neg()?.exp()? + 1.0)?.recip()
}

fn rms_norm(x: &Tensor, gamma: &Tensor) -> candle_core::Result<Tensor> {
    let length = x
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?
        .maximum(1e-12_f32)?;
    (x.broadcast_div(&length)? * (gamma.elem_count() as f64).sqrt())?.broadcast_mul(gamma)
}

// PyTorch gate order is i, f, g, o. Precompute the input projection for all
// timesteps; hidden/cell state still advances one step at a time in each direction.
fn lstm_direction(
    x: &Tensor,
    weights: &LstmDirection,
    tensors: &BTreeMap<String, Tensor>,
    reverse: bool,
) -> Result<Tensor> {
    let (sequence, batch, features) = x.dims3()?;
    let hidden = tensors[&weights.weight_hh].dim(1)?;
    let input = x
        .reshape((sequence * batch, features))?
        .matmul(&tensors[&weights.weight_ih].t()?)?
        .broadcast_add(&tensors[&weights.bias_ih])?
        .reshape((sequence, batch, 4 * hidden))?;
    let recurrent = tensors[&weights.weight_hh].t()?;
    let mut h = Tensor::zeros((batch, hidden), x.dtype(), x.device())?;
    let mut c = h.clone();
    let mut output = Vec::with_capacity(sequence);
    for index in 0..sequence {
        let t = if reverse { sequence - 1 - index } else { index };
        let gates = (input.narrow(0, t, 1)?.squeeze(0)?
            + h.matmul(&recurrent)?
                .broadcast_add(&tensors[&weights.bias_hh])?)?;
        let i = sigmoid(&gates.narrow(1, 0, hidden)?)?;
        let f = sigmoid(&gates.narrow(1, hidden, hidden)?)?;
        let g = gates.narrow(1, 2 * hidden, hidden)?.tanh()?;
        let o = sigmoid(&gates.narrow(1, 3 * hidden, hidden)?)?;
        c = ((f * c)? + (i * g)?)?;
        h = (o * c.tanh()?)?;
        output.push(h.clone());
    }
    if reverse {
        output.reverse();
    }
    Ok(Tensor::stack(&output, 0)?)
}

fn rotate(x: Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, heads, sequence, dim) = x.dims4()?;
    let pairs = x.reshape((batch, heads, sequence, dim / 2, 2))?;
    let even = pairs.narrow(4, 0, 1)?.squeeze(4)?;
    let odd = pairs.narrow(4, 1, 1)?.squeeze(4)?;
    let a = (even.broadcast_mul(cos)? - odd.broadcast_mul(sin)?)?;
    let b = (odd.broadcast_mul(cos)? + even.broadcast_mul(sin)?)?;
    Ok(Tensor::stack(&[a, b], 4)?.reshape((batch, heads, sequence, dim))?)
}

fn attention(
    x: &Tensor,
    step: &Step,
    tensors: &BTreeMap<String, Tensor>,
    rotary: &(Tensor, Tensor),
) -> Result<Tensor> {
    let Step::RoformerAttention {
        heads,
        norm,
        qkv,
        gates_weight,
        gates_bias,
        out,
        ..
    } = step
    else {
        unreachable!()
    };
    let (batch, sequence, dim) = x.dims3()?;
    let heads = *heads;
    let head_dim = tensors[qkv].dim(0)? / (3 * heads);
    let normalized = rms_norm(x, &tensors[norm])?.reshape((batch * sequence, dim))?;
    let projected = normalized
        .matmul(&tensors[qkv].t()?)?
        .reshape((batch, sequence, 3, heads, head_dim))?;
    let component = |index| {
        projected
            .narrow(2, index, 1)?
            .squeeze(2)?
            .transpose(1, 2)?
            .contiguous()
    };
    let q = rotate(component(0)?, &rotary.0, &rotary.1)?;
    let k = rotate(component(1)?, &rotary.0, &rotary.1)?;
    let v = component(2)?;
    let scores = (q.matmul(&k.transpose(2, 3)?)? * (head_dim as f64).powf(-0.5))?;
    let exp = scores.broadcast_sub(&scores.max_keepdim(3)?)?.exp()?;
    let probabilities = exp.broadcast_div(&exp.sum_keepdim(3)?)?;
    let gates = normalized
        .matmul(&tensors[gates_weight].t()?)?
        .broadcast_add(&tensors[gates_bias])?
        .reshape((batch, sequence, heads))?
        .transpose(1, 2)?
        .unsqueeze(3)?;
    let values = probabilities.matmul(&v)?.broadcast_mul(&sigmoid(&gates)?)?;
    Ok(values
        .transpose(1, 2)?
        .contiguous()?
        .reshape((batch * sequence, heads * head_dim))?
        .matmul(&tensors[out].t()?)?
        .reshape((batch, sequence, dim))?)
}
