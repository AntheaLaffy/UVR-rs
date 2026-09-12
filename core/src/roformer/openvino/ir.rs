//! Fixed 1296 graph, expressed in OpenVINO IR v11 without a Python converter.
//! Operation versions and attributes follow the 2026.3.1 reference conversion.

use std::{fmt::Write as _, path::Path};

use anyhow::{Context, Result, ensure};

use super::super::{BANDS, DIM, HEAD_DIM, HEADS};
use crate::{checkpoint::Loader, task::TaskCancelled};

/// A fixed-shape graph built from all 699 original checkpoint tensors.
/// The application passes these buffers to the native runtime in memory.
pub struct RoformerIr {
    pub xml: Vec<u8>,
    pub weights: Vec<u8>,
    pub checkpoint_sha256: String,
    pub frames: usize,
    pub batch: usize,
    pub operations: usize,
}

#[derive(Clone)]
struct Port {
    layer: usize,
    port: usize,
    shape: Vec<usize>,
    precision: &'static str,
}

struct Graph {
    layers: String,
    edges: String,
    count: usize,
    constants: Vec<u8>,
    offset: usize,
    weights: Loader,
}

fn dimensions(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn port_xml(target: &mut String, id: usize, shape: &[usize], precision: &str) {
    write!(target, "<port id=\"{id}\" precision=\"{precision}\">").unwrap();
    for dimension in shape {
        write!(target, "<dim>{dimension}</dim>").unwrap();
    }
    target.push_str("</port>\n");
}

impl Graph {
    fn op(
        &mut self,
        kind: &str,
        version: usize,
        attributes: &str,
        inputs: &[&Port],
        shape: &[usize],
        precision: &'static str,
    ) -> Port {
        let layer = self.count;
        self.count += 1;
        writeln!(self.layers,
            "<layer id=\"{layer}\" name=\"{kind}_{layer}\" type=\"{kind}\" version=\"opset{version}\">").unwrap();
        if !attributes.is_empty() {
            writeln!(self.layers, "<data {attributes} />").unwrap();
        }
        if !inputs.is_empty() {
            self.layers.push_str("<input>\n");
            for (index, input) in inputs.iter().enumerate() {
                port_xml(&mut self.layers, index, &input.shape, input.precision);
                writeln!(self.edges,
                    "<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{layer}\" to-port=\"{index}\" />",
                    input.layer, input.port).unwrap();
            }
            self.layers.push_str("</input>\n");
        }
        if kind != "Result" {
            self.layers.push_str("<output>\n");
            port_xml(&mut self.layers, inputs.len(), shape, precision);
            self.layers.push_str("</output>\n");
        }
        self.layers.push_str("</layer>\n");
        Port {
            layer,
            port: inputs.len(),
            shape: shape.into(),
            precision,
        }
    }

    fn constant(
        &mut self,
        bytes: &[u8],
        shape: &[usize],
        element: &str,
        precision: &'static str,
    ) -> Result<Port> {
        self.constants.extend_from_slice(bytes);
        let attributes = format!(
            "element_type=\"{element}\" shape=\"{}\" offset=\"{}\" size=\"{}\"",
            dimensions(shape),
            self.offset,
            bytes.len()
        );
        self.offset += bytes.len();
        Ok(self.op("Const", 1, &attributes, &[], shape, precision))
    }

    fn integers(&mut self, values: &[i64]) -> Result<Port> {
        self.constant(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            &[values.len()],
            "i64",
            "I64",
        )
    }

    fn floats(&mut self, values: &[f32], shape: &[usize]) -> Result<Port> {
        ensure!(
            values.len() == shape.iter().product::<usize>(),
            "constant shape mismatch"
        );
        self.constant(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
            shape,
            "f32",
            "FP32",
        )
    }

    fn weight_values(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.weights
            .float(name, shape)?
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("{error:?}"))
    }

    fn weight(&mut self, name: &str, shape: &[usize]) -> Result<Port> {
        let values = self.weight_values(name, shape)?;
        self.floats(&values, shape)
    }

    fn unary(&mut self, kind: &str, x: &Port) -> Port {
        self.op(kind, 1, "", &[x], &x.shape, "FP32")
    }

    // The output shape is explicit so each generated edge remains reviewable.
    fn binary(&mut self, kind: &str, a: &Port, b: &Port, shape: &[usize]) -> Port {
        self.op(kind, 1, "auto_broadcast=\"numpy\"", &[a, b], shape, "FP32")
    }

    fn reshape(&mut self, x: &Port, shape: &[usize]) -> Result<Port> {
        ensure!(
            x.shape.iter().product::<usize>() == shape.iter().product::<usize>(),
            "invalid reshape"
        );
        let pattern = self.integers(&shape.iter().map(|&v| v as i64).collect::<Vec<_>>())?;
        Ok(self.op(
            "Reshape",
            1,
            "special_zero=\"false\"",
            &[x, &pattern],
            shape,
            "FP32",
        ))
    }

    fn transpose(&mut self, x: &Port, order: &[usize]) -> Result<Port> {
        let indices = self.integers(&order.iter().map(|&v| v as i64).collect::<Vec<_>>())?;
        let shape: Vec<_> = order.iter().map(|&i| x.shape[i]).collect();
        Ok(self.op("Transpose", 1, "", &[x, &indices], &shape, "FP32"))
    }

    fn slice(
        &mut self,
        x: &Port,
        axis: usize,
        start: usize,
        end: usize,
        step: usize,
    ) -> Result<Port> {
        ensure!(
            start < end && end <= x.shape[axis] && step > 0,
            "invalid slice"
        );
        let starts = self.integers(&[start as i64])?;
        let stops = self.integers(&[end as i64])?;
        let steps = self.integers(&[step as i64])?;
        let axes = self.integers(&[axis as i64])?;
        let mut shape = x.shape.clone();
        shape[axis] = (end - start).div_ceil(step);
        Ok(self.op(
            "Slice",
            8,
            "",
            &[x, &starts, &stops, &steps, &axes],
            &shape,
            "FP32",
        ))
    }

    fn concat(&mut self, inputs: &[Port], axis: usize) -> Result<Port> {
        ensure!(!inputs.is_empty(), "empty concat");
        let mut shape = inputs[0].shape.clone();
        shape[axis] = inputs.iter().map(|p| p.shape[axis]).sum();
        let ports: Vec<_> = inputs.iter().collect();
        Ok(self.op(
            "Concat",
            1,
            &format!("axis=\"{axis}\""),
            &ports,
            &shape,
            "FP32",
        ))
    }

    fn split(&mut self, x: &Port, axis: usize, parts: usize) -> Result<Vec<Port>> {
        ensure!(
            parts > 0 && axis < x.shape.len() && x.shape[axis].is_multiple_of(parts),
            "invalid equal split"
        );
        let axis_input = self.constant(&(axis as i64).to_le_bytes(), &[], "i64", "I64")?;
        let layer = self.count;
        self.count += 1;
        writeln!(self.layers,
            "<layer id=\"{layer}\" name=\"Split_{layer}\" type=\"Split\" version=\"opset1\">\n<data num_splits=\"{parts}\" />\n<input>").unwrap();
        for (index, input) in [x, &axis_input].into_iter().enumerate() {
            port_xml(&mut self.layers, index, &input.shape, input.precision);
            writeln!(self.edges,
                "<edge from-layer=\"{}\" from-port=\"{}\" to-layer=\"{layer}\" to-port=\"{index}\" />",
                input.layer, input.port).unwrap();
        }
        self.layers.push_str("</input>\n<output>\n");
        let mut shape = x.shape.clone();
        shape[axis] /= parts;
        let mut outputs = Vec::with_capacity(parts);
        for index in 0..parts {
            port_xml(&mut self.layers, index + 2, &shape, x.precision);
            outputs.push(Port {
                layer,
                port: index + 2,
                shape: shape.clone(),
                precision: x.precision,
            });
        }
        self.layers.push_str("</output>\n</layer>\n");
        Ok(outputs)
    }

    fn linear(&mut self, x: &Port, prefix: &str, output: usize, bias: bool) -> Result<Port> {
        let input = *x.shape.last().context("scalar linear input")?;
        let weight = self.weight(&format!("{prefix}.weight"), &[output, input])?;
        let mut shape = x.shape.clone();
        *shape.last_mut().unwrap() = output;
        let product = self.op(
            "MatMul",
            1,
            "transpose_a=\"false\" transpose_b=\"true\"",
            &[x, &weight],
            &shape,
            "FP32",
        );
        if bias {
            let bias = self.weight(&format!("{prefix}.bias"), &[output])?;
            Ok(self.binary("Add", &product, &bias, &shape))
        } else {
            Ok(product)
        }
    }

    fn norm(&mut self, x: &Port, name: &str) -> Result<Port> {
        let width = *x.shape.last().context("scalar normalization")?;
        let axes = self.integers(&[-1])?;
        let mut reduced_shape = x.shape.clone();
        *reduced_shape.last_mut().unwrap() = 1;
        let norm = self.op(
            "ReduceL2",
            4,
            "keep_dims=\"true\"",
            &[x, &axes],
            &reduced_shape,
            "FP32",
        );
        let epsilon = self.floats(&[1e-12], &[])?;
        let divisor = self.binary("Maximum", &norm, &epsilon, &reduced_shape);
        let normalized = self.binary("Divide", x, &divisor, &x.shape);
        // Retain F.normalize -> scale -> gamma arithmetic order from the model.
        let scale = self.floats(&[(width as f64).sqrt() as f32], &[])?;
        let scaled = self.binary("Multiply", &normalized, &scale, &x.shape);
        let gamma = self.weight(name, &[width])?;
        Ok(self.binary("Multiply", &scaled, &gamma, &x.shape))
    }

    fn rotary(&mut self, x: &Port, cos: &Port, sin: &Port) -> Result<Port> {
        // Pairwise rotation: [-odd, even], preserving full attention sequence.
        // Split avoids the CPU Slice executor's per-element source/destination
        // index tables for a step-2 last dimension.
        let pairs = self.reshape(x, &[x.shape[0], HEADS, x.shape[2], HEAD_DIM / 2, 2])?;
        let halves = self.split(&pairs, 4, 2)?;
        let negative_odd = self.unary("Negative", &halves[1]);
        let pairs = self.concat(&[negative_odd, halves[0].clone()], 4)?;
        let rotated = self.reshape(&pairs, &x.shape)?;
        let direct = self.binary("Multiply", x, cos, &x.shape);
        let rotated = self.binary("Multiply", &rotated, sin, &x.shape);
        Ok(self.binary("Add", &direct, &rotated, &x.shape))
    }

    fn transformer(&mut self, x: &Port, prefix: &str) -> Result<Port> {
        let (batches, sequence) = (x.shape[0], x.shape[1]);
        let attention = format!("{prefix}.0");
        let normalized = self.norm(x, &format!("{attention}.norm.gamma"))?;
        let qkv = self.linear(&normalized, &format!("{attention}.to_qkv"), 3 * DIM, false)?;
        let frequencies =
            self.weight_values(&format!("{attention}.rotary_embed.freqs"), &[HEAD_DIM / 2])?;
        let mut cos = Vec::with_capacity(sequence * HEAD_DIM);
        let mut sin = Vec::with_capacity(sequence * HEAD_DIM);
        for position in 0..sequence {
            for &frequency in &frequencies {
                let angle = position as f32 * frequency;
                cos.extend([angle.cos(); 2]);
                sin.extend([angle.sin(); 2]);
            }
        }
        let cos = self.floats(&cos, &[1, 1, sequence, HEAD_DIM])?;
        let sin = self.floats(&sin, &[1, 1, sequence, HEAD_DIM])?;
        let mut components = Vec::with_capacity(3);
        for component in 0..3 {
            let part = self.slice(&qkv, 2, component * DIM, (component + 1) * DIM, 1)?;
            let part = self.reshape(&part, &[batches, sequence, HEADS, HEAD_DIM])?;
            let part = self.transpose(&part, &[0, 2, 1, 3])?;
            components.push(if component < 2 {
                self.rotary(&part, &cos, &sin)?
            } else {
                part
            });
        }
        let attended = self.op(
            "ScaledDotProductAttention",
            13,
            "causal=\"false\"",
            &[&components[0], &components[1], &components[2]],
            &components[0].shape,
            "FP32",
        );
        let gates = self.linear(&normalized, &format!("{attention}.to_gates"), HEADS, true)?;
        let gates = self.unary("Sigmoid", &gates);
        let gates = self.reshape(&gates, &[batches, sequence, HEADS, 1])?;
        let gates = self.transpose(&gates, &[0, 2, 1, 3])?;
        let attended = self.binary("Multiply", &attended, &gates, &attended.shape);
        let attended = self.transpose(&attended, &[0, 2, 1, 3])?;
        let attended = self.reshape(&attended, &x.shape)?;
        let attention = self.linear(&attended, &format!("{attention}.to_out.0"), DIM, false)?;
        let residual = self.binary("Add", &attention, x, &x.shape);
        let feed = self.norm(&residual, &format!("{prefix}.1.net.0.gamma"))?;
        let feed = self.linear(&feed, &format!("{prefix}.1.net.1"), 4 * DIM, true)?;
        let feed = self.op(
            "Gelu",
            7,
            "approximation_mode=\"ERF\"",
            &[&feed],
            &feed.shape,
            "FP32",
        );
        let feed = self.linear(&feed, &format!("{prefix}.1.net.4"), DIM, true)?;
        Ok(self.binary("Add", &feed, &residual, &x.shape))
    }

    fn network(
        &mut self,
        batch: usize,
        frames: usize,
        keep_going: &mut impl FnMut() -> bool,
    ) -> Result<()> {
        let input = self.op(
            "Parameter",
            1,
            &format!("shape=\"{batch},{frames},4100\" element_type=\"f32\""),
            &[],
            &[batch, frames, 4100],
            "FP32",
        );
        let mut bands = Vec::with_capacity(BANDS.len());
        let mut first = 0;
        for (index, &bins) in BANDS.iter().enumerate() {
            if !keep_going() {
                return Err(TaskCancelled.into());
            }
            let band = self.slice(&input, 2, first, first + bins * 4, 1)?;
            first += bins * 4;
            let band = self.norm(&band, &format!("band_split.to_features.{index}.0.gamma"))?;
            let band = self.linear(
                &band,
                &format!("band_split.to_features.{index}.1"),
                DIM,
                true,
            )?;
            bands.push(self.reshape(&band, &[batch, frames, 1, DIM])?);
        }
        let mut x = self.concat(&bands, 2)?;
        for depth in 0..12 {
            if !keep_going() {
                return Err(TaskCancelled.into());
            }
            x = self.transpose(&x, &[0, 2, 1, 3])?;
            x = self.reshape(&x, &[batch * BANDS.len(), frames, DIM])?;
            x = self.transformer(&x, &format!("layers.{depth}.0.layers.0"))?;
            x = self.reshape(&x, &[batch, BANDS.len(), frames, DIM])?;
            x = self.transpose(&x, &[0, 2, 1, 3])?;
            x = self.reshape(&x, &[batch * frames, BANDS.len(), DIM])?;
            x = self.transformer(&x, &format!("layers.{depth}.1.layers.0"))?;
            x = self.reshape(&x, &[batch, frames, BANDS.len(), DIM])?;
        }
        x = self.norm(&x, "final_norm.gamma")?;
        let mut masks = Vec::with_capacity(BANDS.len());
        for (index, &bins) in BANDS.iter().enumerate() {
            if !keep_going() {
                return Err(TaskCancelled.into());
            }
            let band = self.slice(&x, 2, index, index + 1, 1)?;
            let band = self.reshape(&band, &[batch, frames, DIM])?;
            let prefix = format!("mask_estimators.0.to_freqs.{index}.0");
            let band = self.linear(&band, &format!("{prefix}.0"), 4 * DIM, true)?;
            let band = self.unary("Tanh", &band);
            let band = self.linear(&band, &format!("{prefix}.2"), 8 * bins, true)?;
            let values = self.slice(&band, 2, 0, 4 * bins, 1)?;
            let gates = self.slice(&band, 2, 4 * bins, 8 * bins, 1)?;
            let gates = self.unary("Sigmoid", &gates);
            masks.push(self.binary("Multiply", &values, &gates, &values.shape));
        }
        let mask = self.concat(&masks, 2)?;
        self.op("Result", 1, "", &[&mask], &mask.shape, "FP32");
        Ok(())
    }
}

impl RoformerIr {
    /// Builds an IR in memory, strictly validating checkpoint identity and keys.
    pub fn from_checkpoint(
        checkpoint: &Path,
        frames: usize,
        keep_going: impl FnMut() -> bool,
    ) -> Result<Self> {
        Self::from_checkpoint_with_batch(checkpoint, 1, frames, keep_going)
    }

    /// Builds a fixed batch graph for independent windows. The product keeps
    /// batch one; larger batches are used by throughput experiments.
    pub fn from_checkpoint_with_batch(
        checkpoint: &Path,
        batch: usize,
        frames: usize,
        mut keep_going: impl FnMut() -> bool,
    ) -> Result<Self> {
        ensure!((1..=8).contains(&batch), "batch must be 1..8");
        ensure!((4..=801).contains(&frames), "frame count must be 4..801");
        if !keep_going() {
            return Err(TaskCancelled.into());
        }
        let identity = crate::weights::fingerprint(checkpoint)?;
        ensure!(
            identity.sha256 == "f6c94864adfb73bbb0ca58ec14d58dd0b364549e9fb61433ae51916f3e2f8d0b",
            "unsupported original checkpoint"
        );
        if !keep_going() {
            return Err(TaskCancelled.into());
        }
        // The fixed weight set dominates storage; reserve room for position constants
        // to avoid reallocating and copying hundreds of MiB during construction.
        let capacity = usize::try_from(identity.size_bytes)?
            .checked_add(8 * 1024 * 1024)
            .context("IR size overflow")?;
        let mut graph = Graph {
            layers: String::new(),
            edges: String::new(),
            count: 0,
            constants: Vec::with_capacity(capacity),
            offset: 0,
            weights: Loader::new(checkpoint)?,
        };
        graph.network(batch, frames, &mut keep_going)?;
        graph.weights.finish(699)?;
        let xml = format!(
            "<?xml version=\"1.0\"?>\n<net name=\"1296_rust\" version=\"11\">\n<layers>\n{}</layers>\n<edges>\n{}</edges>\n</net>\n",
            graph.layers, graph.edges
        ).into_bytes();
        Ok(Self {
            xml,
            weights: graph.constants,
            checkpoint_sha256: identity.sha256,
            frames,
            batch,
            operations: graph.count,
        })
    }
}
