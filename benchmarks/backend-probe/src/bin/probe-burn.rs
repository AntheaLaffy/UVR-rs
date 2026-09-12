use std::collections::BTreeMap;

use anyhow::Result;
use burn_flex::Flex;
use burn_tensor::{
    Tensor, TensorData, activation, module,
    ops::{ConvOptions, InterpolateMode, InterpolateOptions},
};
use uvr_backend_probe::{LstmDirection, Probe, Step, measure, rotary_tables};

type T4 = Tensor<Flex, 4>;
#[path = "../../../../core/src/roformer/cpu.rs"]
mod roformer_cpu;

#[derive(Clone, Copy)]
struct Options {
    flattened: bool,
    parallel_gelu: bool,
    fused_norm: bool,
    fused_rope: bool,
}

fn main() -> Result<()> {
    let probe = Probe::load()?;
    let flattened = match std::env::var("UVR_LINEAR_LAYOUT").as_deref() {
        Ok("flattened") => true,
        Ok("batched") | Err(_) => false,
        Ok(value) => anyhow::bail!("invalid UVR_LINEAR_LAYOUT: {value}"),
    };
    let prefix = std::env::var("UVR_PROBE_CASE_PREFIX").unwrap_or_default();
    let flag = |name| -> Result<bool> {
        Ok(match std::env::var(name).as_deref() {
            Ok("1") => true,
            Ok("0") | Err(_) => false,
            Ok(value) => anyhow::bail!("invalid {name}: {value}"),
        })
    };
    let options = Options {
        flattened,
        parallel_gelu: flag("UVR_PARALLEL_GELU")?,
        fused_norm: flag("UVR_FUSED_RMS_NORM")?,
        fused_rope: flag("UVR_FUSED_ROPE")?,
    };
    let device = Default::default();
    let tensors: BTreeMap<String, Tensor<Flex, 4>> = probe
        .tensors
        .iter()
        .map(|(name, data)| {
            let shape = match data.shape.as_slice() {
                [channels] => [1, *channels, 1, 1],
                [rows, columns] => [1, 1, *rows, *columns],
                [a, b, c] => [1, *a, *b, *c],
                [a, b, c, d] => [*a, *b, *c, *d],
                _ => panic!("expected rank 1 through 4"),
            };
            (
                name.clone(),
                Tensor::from_data(TensorData::new(data.values.clone(), shape), &device),
            )
        })
        .collect();
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
                    Tensor::<Flex, 4>::from_data(TensorData::new(data.values, data.shape), &device)
                };
                rotary.insert(case.name.clone(), (tensor(cos), tensor(sin)));
            }
        }
    }
    let mut measurements = Vec::new();
    for case in &probe.cases {
        if !case.name.starts_with(&prefix) {
            continue;
        }
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
                    } => module::conv2d(
                        x,
                        tensors[weight].clone(),
                        None,
                        ConvOptions::new([*stride; 2], [*padding; 2], [*dilation; 2], *groups),
                    ),
                    Step::BatchNorm {
                        weight,
                        bias,
                        mean,
                        variance,
                        epsilon,
                    } => {
                        (x - tensors[mean].clone()) / (tensors[variance].clone() + *epsilon).sqrt()
                            * tensors[weight].clone()
                            + tensors[bias].clone()
                    }
                    Step::LeakyRelu { slope } => activation::leaky_relu(x, f64::from(*slope)),
                    Step::Bilinear {
                        size,
                        align_corners,
                    } => module::interpolate(
                        x,
                        *size,
                        InterpolateOptions::new(InterpolateMode::Bilinear)
                            .with_align_corners(*align_corners),
                    ),
                    Step::BiLstm { forward, reverse } => Tensor::cat(
                        vec![
                            lstm_direction(x.clone(), forward, &tensors, false),
                            lstm_direction(x, reverse, &tensors, true),
                        ],
                        3,
                    ),
                    Step::RoformerAttention { .. } => {
                        attention(x, step, &tensors, &rotary[&case.name], options)
                    }
                    Step::RoformerFeedForward {
                        norm,
                        input_weight,
                        input_bias,
                        output_weight,
                        output_bias,
                    } => {
                        let hidden_dim = tensors[input_weight].dims()[2];
                        let dim = x.dims()[3];
                        let hidden = linear(
                            rms_norm(x, tensors[norm].clone(), options.fused_norm),
                            tensors[input_weight].clone().transpose(),
                            flattened,
                        ) + tensors[input_bias].clone().reshape([1, 1, 1, hidden_dim]);
                        let hidden = if options.parallel_gelu {
                            roformer_cpu::gelu(hidden)
                        } else {
                            activation::gelu(hidden)
                        };
                        linear(
                            hidden,
                            tensors[output_weight].clone().transpose(),
                            flattened,
                        ) + tensors[output_bias].clone().reshape([1, 1, 1, dim])
                    }
                };
            }
            let shape = if probe.tensors[&case.input].shape.len() == 3 {
                x.dims()[1..].to_vec()
            } else {
                x.dims().to_vec()
            };
            Ok((shape, x.into_data().to_vec::<f32>()?))
        })?);
    }
    anyhow::ensure!(!measurements.is_empty(), "no matching cases");
    let backend = format!(
        "burn-flex 0.21.0 CPU, {} linear, {} GELU, fused RMSNorm={}, fused RoPE={}",
        if flattened { "flattened" } else { "batched" },
        if options.parallel_gelu {
            "parallel"
        } else {
            "serial"
        },
        options.fused_norm,
        options.fused_rope
    );
    probe.write(&backend, measurements)
}

fn linear(x: Tensor<Flex, 4>, weight: Tensor<Flex, 4>, flattened: bool) -> Tensor<Flex, 4> {
    if !flattened {
        return x.matmul(weight);
    }
    let [a, b, c, input] = x.dims();
    let output = weight.dims()[3];
    x.reshape([a * b * c, input])
        .matmul(weight.reshape([input, output]))
        .reshape([a, b, c, output])
}

fn rms_norm(x: Tensor<Flex, 4>, gamma: Tensor<Flex, 4>, fused: bool) -> Tensor<Flex, 4> {
    if fused {
        return roformer_cpu::rms_norm(x, gamma);
    }
    let dim = x.dims()[3];
    let length = (x.clone() * x.clone()).sum_dim(3).sqrt().clamp_min(1e-12);
    x / length * (dim as f32).sqrt() * gamma.reshape([1, 1, 1, dim])
}

fn lstm_direction(
    x: Tensor<Flex, 4>,
    weights: &LstmDirection,
    tensors: &BTreeMap<String, Tensor<Flex, 4>>,
    reverse: bool,
) -> Tensor<Flex, 4> {
    let [_, sequence, batch, features] = x.dims();
    let hidden = tensors[&weights.weight_hh].dims()[3];
    let input = x.reshape([sequence * batch, features]).matmul(
        tensors[&weights.weight_ih]
            .clone()
            .reshape([4 * hidden, features])
            .transpose(),
    ) + tensors[&weights.bias_ih].clone().reshape([1, 4 * hidden]);
    let recurrent = tensors[&weights.weight_hh]
        .clone()
        .reshape([4 * hidden, hidden])
        .transpose();
    let bias = tensors[&weights.bias_hh].clone().reshape([1, 4 * hidden]);
    let mut h = Tensor::<Flex, 2>::zeros([batch, hidden], &input.device());
    let mut c = h.clone();
    let mut output = Vec::with_capacity(sequence);
    for index in 0..sequence {
        let t = if reverse { sequence - 1 - index } else { index };
        let gates = input
            .clone()
            .slice([t * batch..(t + 1) * batch, 0..4 * hidden])
            + (h.matmul(recurrent.clone()) + bias.clone());
        let i = activation::sigmoid(gates.clone().slice([0..batch, 0..hidden]));
        let f = activation::sigmoid(gates.clone().slice([0..batch, hidden..2 * hidden]));
        let g = gates
            .clone()
            .slice([0..batch, 2 * hidden..3 * hidden])
            .tanh();
        let o = activation::sigmoid(gates.slice([0..batch, 3 * hidden..4 * hidden]));
        c = f * c + i * g;
        h = o * c.clone().tanh();
        output.push(h.clone());
    }
    if reverse {
        output.reverse();
    }
    Tensor::cat(output, 0).reshape([1, sequence, batch, hidden])
}

fn rotate(
    x: Tensor<Flex, 4>,
    cos: Tensor<Flex, 4>,
    sin: Tensor<Flex, 4>,
    fused: bool,
) -> Tensor<Flex, 4> {
    if fused {
        return roformer_cpu::rotate(x, cos, sin);
    }
    let [batch, heads, sequence, dim] = x.dims();
    let pairs = x.reshape([batch, heads, sequence, dim / 2, 2]);
    let even = pairs.clone().slice_dim(4, 0..1).squeeze_dim::<4>(4);
    let odd = pairs.slice_dim(4, 1..2).squeeze_dim::<4>(4);
    let a = even.clone() * cos.clone() - odd.clone() * sin.clone();
    let b = odd * cos + even * sin;
    Tensor::stack::<5>(vec![a, b], 4).reshape([batch, heads, sequence, dim])
}

fn attention(
    x: Tensor<Flex, 4>,
    step: &Step,
    tensors: &BTreeMap<String, Tensor<Flex, 4>>,
    rotary: &(Tensor<Flex, 4>, Tensor<Flex, 4>),
    options: Options,
) -> Tensor<Flex, 4> {
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
    let [_, batch, sequence, _] = x.dims();
    let heads = *heads;
    let head_dim = tensors[qkv].dims()[2] / (3 * heads);
    let width = heads * head_dim;
    let normalized = rms_norm(x, tensors[norm].clone(), options.fused_norm);
    let projected = linear(
        normalized.clone(),
        tensors[qkv].clone().transpose(),
        options.flattened,
    );
    let component = |index| {
        projected
            .clone()
            .slice_dim(3, index * width..(index + 1) * width)
            .reshape([batch, sequence, heads, head_dim])
            .swap_dims(1, 2)
    };
    let q = rotate(
        component(0),
        rotary.0.clone(),
        rotary.1.clone(),
        options.fused_rope,
    );
    let k = rotate(
        component(1),
        rotary.0.clone(),
        rotary.1.clone(),
        options.fused_rope,
    );
    let scores = q.matmul(k.transpose()) * (head_dim as f32).powf(-0.5);
    let probabilities = activation::softmax(scores, 3);
    let gates = (linear(
        normalized,
        tensors[gates_weight].clone().transpose(),
        options.flattened,
    ) + tensors[gates_bias].clone().reshape([1, 1, 1, heads]))
    .reshape([batch, sequence, heads, 1])
    .swap_dims(1, 2);
    let values = probabilities.matmul(component(2)) * activation::sigmoid(gates);
    linear(
        values.swap_dims(1, 2).reshape([1, batch, sequence, width]),
        tensors[out].clone().transpose(),
        options.flattened,
    )
}
