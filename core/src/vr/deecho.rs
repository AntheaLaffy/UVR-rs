//! Fixed CascadedNet 218409 from UVR `5517e0cf`, nets_new/layers_new.py.

use super::*;

type T2 = Tensor<Flex, 2>;
type T3 = Tensor<Flex, 3>;

/// CPU FP32 mask network for UVR-DeEcho-DeReverb.pth.
pub struct DeEchoModel {
    low1: RecurrentNet,
    low1_out: Block,
    high1: RecurrentNet,
    low2: RecurrentNet,
    low2_out: Block,
    high2: RecurrentNet,
    full3: RecurrentNet,
    out: Conv,
}

impl DeEchoModel {
    pub const BINS: usize = 672;
    pub const OFFSET: usize = 64;

    pub fn load(path: &Path) -> Result<Self> {
        let identity = crate::weights::fingerprint(path)?;
        ensure!(
            identity.sha256 == "e644028ec82865dc0fe082bc6fea85a43f7c71cfe375caee2da2d154aa661ee7",
            "unsupported DeEcho-DeReverb checkpoint SHA-256: {}",
            identity.sha256
        );
        let mut loader = Loader::new(path)?;
        let model = Self {
            low1: RecurrentNet::load(&mut loader, "stg1_low_band_net.0", 2, 32, 168, 64)?,
            low1_out: Block::load(
                &mut loader,
                "stg1_low_band_net.1.conv",
                32,
                16,
                1,
                1,
                0,
                false,
            )?,
            high1: RecurrentNet::load(&mut loader, "stg1_high_band_net", 2, 16, 168, 32)?,
            low2: RecurrentNet::load(&mut loader, "stg2_low_band_net.0", 18, 64, 168, 64)?,
            low2_out: Block::load(
                &mut loader,
                "stg2_low_band_net.1.conv",
                64,
                32,
                1,
                1,
                0,
                false,
            )?,
            high2: RecurrentNet::load(&mut loader, "stg2_high_band_net", 18, 32, 168, 32)?,
            full3: RecurrentNet::load(&mut loader, "stg3_full_band_net", 50, 64, 336, 64)?,
            out: Conv::load(&mut loader, "out.weight", [2, 64, 1, 1], 1, 0, 1, 1)?,
        };
        loader.float("aux_out.weight", &[2, 48, 1, 1])?;
        loader.finish(689)?;
        Ok(model)
    }

    /// Channel/frequency/time-major input `[2, 673, frames]`, with 64 context
    /// frames on each side. Returns the cropped mask `[2, 673, frames - 128]`.
    pub fn predict_mask(&self, magnitude: &[f32], frames: usize) -> Result<Vec<f32>> {
        let x = input_window(magnitude, frames, Self::BINS, Self::OFFSET)?;
        let low_in = x.clone().slice_dim(2, 0..Self::BINS / 2);
        let high_in = x.clone().slice_dim(2, Self::BINS / 2..Self::BINS);
        let low1 = self.low1_out.forward(self.low1.forward(low_in.clone()));
        let high1 = self.high1.forward(high_in.clone());
        let aux1 = T4::cat(vec![low1.clone(), high1.clone()], 2);
        let low2 = self
            .low2_out
            .forward(self.low2.forward(T4::cat(vec![low_in, low1], 1)));
        let high2 = self.high2.forward(T4::cat(vec![high_in, high1], 1));
        let aux2 = T4::cat(vec![low2, high2], 2);
        let full = self.full3.forward(T4::cat(vec![x, aux1, aux2], 1));
        finish_mask(
            activation::sigmoid(self.out.forward(full)),
            frames,
            Self::BINS,
            Self::OFFSET,
        )
    }
}

struct RecurrentNet {
    first: Block,
    encoders: Vec<(Block, Block)>,
    aspp: DilatedAspp,
    decoders: Vec<Block>,
    recurrent: RecurrentBlock,
    last: Block,
}

impl RecurrentNet {
    fn load(
        loader: &mut Loader,
        prefix: &str,
        input: usize,
        channels: usize,
        frequencies: usize,
        hidden: usize,
    ) -> Result<Self> {
        let first = Block::load(
            loader,
            &format!("{prefix}.enc1.conv"),
            input,
            channels,
            3,
            1,
            1,
            false,
        )?;
        let mut encoders = Vec::new();
        let mut input = channels;
        for (level, multiplier) in [2, 4, 6, 8].into_iter().enumerate() {
            let output = channels * multiplier;
            encoders.push((
                Block::load(
                    loader,
                    &format!("{prefix}.enc{}.conv1.conv", level + 2),
                    input,
                    output,
                    3,
                    2,
                    1,
                    true,
                )?,
                Block::load(
                    loader,
                    &format!("{prefix}.enc{}.conv2.conv", level + 2),
                    output,
                    output,
                    3,
                    1,
                    1,
                    true,
                )?,
            ));
            input = output;
        }
        let aspp = DilatedAspp::load(loader, &format!("{prefix}.aspp"), channels * 8)?;
        let mut decoders = Vec::new();
        for (level, input, output) in [(4, 14, 6), (3, 10, 4), (2, 6, 2)] {
            decoders.push(Block::load(
                loader,
                &format!("{prefix}.dec{level}.conv1.conv"),
                channels * input,
                channels * output,
                3,
                1,
                1,
                false,
            )?);
        }
        Ok(Self {
            first,
            encoders,
            aspp,
            decoders,
            recurrent: RecurrentBlock::load(
                loader,
                &format!("{prefix}.lstm_dec2"),
                channels * 2,
                frequencies,
                hidden,
            )?,
            last: Block::load(
                loader,
                &format!("{prefix}.dec1.conv1.conv"),
                channels * 3 + 1,
                channels,
                3,
                1,
                1,
                false,
            )?,
        })
    }

    fn forward(&self, x: T4) -> T4 {
        let mut x = self.first.forward(x);
        let first_skip = x.clone();
        let mut skips = Vec::with_capacity(4);
        for (down, refine) in &self.encoders {
            x = refine.forward(down.forward(x));
            skips.push(x.clone());
        }
        x = self.aspp.forward(x);
        // The deepest encoder output feeds ASPP; the three shallower outputs
        // are decoder skips. The final skip predates all four downsamplings.
        skips.pop();
        for (decoder, skip) in self.decoders.iter().zip(skips.into_iter().rev()) {
            x = decode(decoder, x, skip);
        }
        let recurrent = self.recurrent.forward(x.clone());
        decode(&self.last, T4::cat(vec![x, recurrent], 1), first_skip)
    }
}

fn decode(block: &Block, x: T4, skip: T4) -> T4 {
    let [_, _, height, width] = x.dims();
    let x = resize(x, height * 2, width * 2);
    let offset = (skip.dims()[3] - width * 2) / 2;
    block.forward(T4::cat(
        vec![x, skip.slice_dim(3, offset..offset + width * 2)],
        1,
    ))
}

struct DilatedAspp {
    pooled: Block,
    direct: Block,
    dilated: [Block; 3],
    bottleneck: Block,
}

impl DilatedAspp {
    fn load(loader: &mut Loader, prefix: &str, channels: usize) -> Result<Self> {
        let dilated = |loader: &mut Loader, index, dilation| -> Result<Block> {
            let prefix = format!("{prefix}.conv{index}.conv");
            Ok(Block {
                conv: Conv {
                    weight: ConvWeight::Direct(T4::from_data(
                        loader.float(&format!("{prefix}.0.weight"), &[channels, channels, 3, 3])?,
                        &Default::default(),
                    )),
                    stride: 1,
                    padding: dilation,
                    dilation,
                    groups: 1,
                },
                norm: Norm::load(loader, &format!("{prefix}.1"), channels)?,
                leaky: false,
            })
        };
        Ok(Self {
            pooled: Block::load(
                loader,
                &format!("{prefix}.conv1.1.conv"),
                channels,
                channels,
                1,
                1,
                0,
                false,
            )?,
            direct: Block::load(
                loader,
                &format!("{prefix}.conv2.conv"),
                channels,
                channels,
                1,
                1,
                0,
                false,
            )?,
            dilated: [
                dilated(loader, 3, [4, 2])?,
                dilated(loader, 4, [8, 4])?,
                dilated(loader, 5, [12, 6])?,
            ],
            bottleneck: Block::load(
                loader,
                &format!("{prefix}.bottleneck.conv"),
                channels * 5,
                channels,
                1,
                1,
                0,
                false,
            )?,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let [_, _, height, width] = x.dims();
        let mut features = vec![
            resize(self.pooled.forward(x.clone().mean_dim(2)), height, width),
            self.direct.forward(x.clone()),
        ];
        for layer in &self.dilated {
            features.push(layer.forward(x.clone()));
        }
        self.bottleneck.forward(T4::cat(features, 1))
    }
}

struct RecurrentBlock {
    conv: Block,
    forward: Direction,
    reverse: Direction,
    dense_weight: T2,
    dense_bias: T2,
    norm: Norm,
}

impl RecurrentBlock {
    fn load(
        loader: &mut Loader,
        prefix: &str,
        channels: usize,
        frequencies: usize,
        hidden: usize,
    ) -> Result<Self> {
        let dense_weight = T2::from_data(
            loader.float(
                &format!("{prefix}.dense.0.weight"),
                &[frequencies, 2 * hidden],
            )?,
            &Default::default(),
        )
        .transpose();
        let mut bias = loader.float(&format!("{prefix}.dense.0.bias"), &[frequencies])?;
        bias.shape = [1, frequencies].into();
        Ok(Self {
            conv: Block::load(
                loader,
                &format!("{prefix}.conv.conv"),
                channels,
                1,
                1,
                1,
                0,
                false,
            )?,
            forward: Direction::load(loader, &format!("{prefix}.lstm"), "", frequencies, hidden)?,
            reverse: Direction::load(
                loader,
                &format!("{prefix}.lstm"),
                "_reverse",
                frequencies,
                hidden,
            )?,
            dense_weight,
            dense_bias: T2::from_data(bias, &Default::default()),
            norm: Norm::load(loader, &format!("{prefix}.dense.1"), frequencies)?,
        })
    }
    fn forward(&self, x: T4) -> T4 {
        let [batch, _, frequencies, frames] = x.dims();
        let x = self.conv.forward(x).squeeze_dim::<3>(1).permute([2, 0, 1]);
        let x = T3::cat(
            vec![
                self.forward.run(x.clone(), false),
                self.reverse.run(x, true),
            ],
            2,
        );
        let hidden = x.dims()[2];
        let x = x
            .reshape([frames * batch, hidden])
            .matmul(self.dense_weight.clone())
            + self.dense_bias.clone();
        let x = x
            .reshape([frames, batch, 1, frequencies])
            .permute([1, 3, 2, 0]);
        self.norm.forward(x, false).swap_dims(1, 2)
    }
}

struct Direction {
    input_weight: T2,
    hidden_weight: T2,
    input_bias: T2,
    hidden_bias: T2,
    hidden: usize,
}

impl Direction {
    fn load(
        loader: &mut Loader,
        prefix: &str,
        suffix: &str,
        features: usize,
        hidden: usize,
    ) -> Result<Self> {
        let input_weight = T2::from_data(
            loader.float(
                &format!("{prefix}.weight_ih_l0{suffix}"),
                &[4 * hidden, features],
            )?,
            &Default::default(),
        )
        .transpose();
        let hidden_weight = T2::from_data(
            loader.float(
                &format!("{prefix}.weight_hh_l0{suffix}"),
                &[4 * hidden, hidden],
            )?,
            &Default::default(),
        )
        .transpose();
        let bias = |loader: &mut Loader, kind| -> Result<T2> {
            let mut data =
                loader.float(&format!("{prefix}.bias_{kind}_l0{suffix}"), &[4 * hidden])?;
            data.shape = [1, 4 * hidden].into();
            Ok(T2::from_data(data, &Default::default()))
        };
        Ok(Self {
            input_weight,
            hidden_weight,
            input_bias: bias(loader, "ih")?,
            hidden_bias: bias(loader, "hh")?,
            hidden,
        })
    }
    fn run(&self, x: T3, reverse: bool) -> T3 {
        let [frames, batch, features] = x.dims();
        let hidden = self.hidden;
        let input = x
            .reshape([frames * batch, features])
            .matmul(self.input_weight.clone())
            + self.input_bias.clone();
        let mut h = T2::zeros([batch, hidden], &input.device());
        let mut c = h.clone();
        let mut output = Vec::with_capacity(frames);
        for index in 0..frames {
            let t = if reverse { frames - 1 - index } else { index };
            let gates = input
                .clone()
                .slice([t * batch..(t + 1) * batch, 0..4 * hidden])
                + (h.matmul(self.hidden_weight.clone()) + self.hidden_bias.clone());
            // PyTorch stores the four gates as i, f, g, o for each direction.
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
        T2::cat(output, 0).reshape([frames, batch, hidden])
    }
}
