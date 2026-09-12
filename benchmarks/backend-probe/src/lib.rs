use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Deserialize)]
struct TensorFile {
    file: String,
    shape: Vec<usize>,
    sha256: String,
}

#[derive(Deserialize)]
struct Manifest {
    tensors: BTreeMap<String, TensorFile>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
pub struct Case {
    pub name: String,
    pub input: String,
    pub expected: String,
    pub steps: Vec<Step>,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Step {
    Conv2d {
        weight: String,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    },
    BatchNorm {
        weight: String,
        bias: String,
        mean: String,
        variance: String,
        epsilon: f32,
    },
    LeakyRelu {
        slope: f32,
    },
    Bilinear {
        size: [usize; 2],
        align_corners: bool,
    },
    BiLstm {
        forward: LstmDirection,
        reverse: LstmDirection,
    },
    RoformerAttention {
        heads: usize,
        norm: String,
        qkv: String,
        gates_weight: String,
        gates_bias: String,
        out: String,
        rotary_frequencies: String,
    },
    RoformerFeedForward {
        norm: String,
        input_weight: String,
        input_bias: String,
        output_weight: String,
        output_bias: String,
    },
}

#[derive(Deserialize)]
pub struct LstmDirection {
    pub weight_ih: String,
    pub weight_hh: String,
    pub bias_ih: String,
    pub bias_hh: String,
}

// Both candidates use the same FP32 rotary cache, constructed in Rust. Cache
// creation belongs to tensor preparation; applying rotation remains timed.
pub fn rotary_tables(frequencies: &Data, sequence: usize) -> (Data, Data) {
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for position in 0..sequence {
        for frequency in &frequencies.values {
            let angle = position as f32 * frequency;
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    let shape = vec![1, 1, sequence, frequencies.values.len()];
    (
        Data {
            shape: shape.clone(),
            values: cos,
        },
        Data { shape, values: sin },
    )
}

pub struct Data {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

pub struct Probe {
    pub tensors: BTreeMap<String, Data>,
    pub cases: Vec<Case>,
    manifest_sha256: String,
    output: PathBuf,
    threads: usize,
}

#[derive(Serialize)]
pub struct Measurement {
    name: String,
    first_execution_seconds: f64,
    warm_seconds: Vec<f64>,
    max_absolute_error: f64,
    rmse: f64,
    shape: Vec<usize>,
}

#[derive(Serialize)]
pub struct Verification {
    name: String,
    max_absolute_error: f64,
    rmse: f64,
    shape: Vec<usize>,
}

impl Probe {
    pub fn load() -> Result<Self> {
        let args: Vec<_> = std::env::args_os().skip(1).collect();
        ensure!(
            args.len() == 2,
            "usage: probe-<backend> <fixture-directory> <report.json>"
        );
        let root = PathBuf::from(&args[0]);
        Self::load_paths(root, PathBuf::from(&args[1]))
    }

    pub fn load_paths(root: PathBuf, output: PathBuf) -> Result<Self> {
        let raw = std::fs::read(root.join("manifest.json"))?;
        let manifest: Manifest = serde_json::from_slice(&raw)?;
        ensure!(!manifest.cases.is_empty(), "empty probe manifest");
        let mut tensors = BTreeMap::new();
        for (name, file) in manifest.tensors {
            let bytes = std::fs::read(root.join(&file.file)).with_context(|| file.file.clone())?;
            ensure!(
                format!("{:x}", Sha256::digest(&bytes)) == file.sha256,
                "checksum mismatch: {name}"
            );
            let elements = file
                .shape
                .iter()
                .try_fold(1_usize, |product, dim| product.checked_mul(*dim))
                .context("shape overflow")?;
            ensure!(
                elements.checked_mul(4) == Some(bytes.len()),
                "invalid tensor size: {name}"
            );
            let values: Vec<_> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite input: {name}"
            );
            tensors.insert(
                name,
                Data {
                    shape: file.shape,
                    values,
                },
            );
        }
        let threads = std::env::var("RAYON_NUM_THREADS")
            .context("set RAYON_NUM_THREADS to a fixed positive budget")?
            .parse()?;
        ensure!(threads > 0, "thread budget must be positive");
        Ok(Self {
            tensors,
            cases: manifest.cases,
            manifest_sha256: format!("{:x}", Sha256::digest(raw)),
            output,
            threads,
        })
    }

    pub fn write(&self, backend: &str, measurements: Vec<Measurement>) -> Result<()> {
        let report = serde_json::json!({
            "backend": backend,
            "threads": self.threads,
            "manifest_sha256": self.manifest_sha256,
            "timing": "execution closure including CPU operations and output materialization; excludes fixture I/O and preparation before the closure",
            "warmup_executions": 1,
            "absolute_tolerance": 1e-3,
            "relative_tolerance": 1e-4,
            "measurements": measurements,
        });
        std::fs::write(&self.output, serde_json::to_string_pretty(&report)? + "\n")?;
        Ok(())
    }

    pub fn write_verified(&self, backend: &str, checks: Vec<Verification>) -> Result<()> {
        let report = serde_json::json!({
            "backend": backend,
            "mode": "verification_only",
            "manifest_sha256": self.manifest_sha256,
            "threads": self.threads,
            "absolute_tolerance": 1e-3,
            "relative_tolerance": 1e-4,
            "checks": checks,
        });
        std::fs::write(&self.output, serde_json::to_string_pretty(&report)? + "\n")?;
        Ok(())
    }
}

pub fn verify(
    name: &str,
    expected: &Data,
    execute: impl FnOnce() -> Result<(Vec<usize>, Vec<f32>)>,
) -> Result<Verification> {
    let (shape, output) = execute()?;
    let (max_absolute_error, rmse) =
        validate(&shape, &output, expected).with_context(|| name.to_owned())?;
    eprintln!("{name}: numerical check passed");
    Ok(Verification {
        name: name.into(),
        max_absolute_error,
        rmse,
        shape,
    })
}

fn validate(shape: &[usize], actual: &[f32], expected: &Data) -> Result<(f64, f64)> {
    ensure!(
        shape == expected.shape && actual.len() == expected.values.len(),
        "output shape mismatch"
    );
    let mut max_error = 0_f64;
    let mut squared_error = 0_f64;
    for (i, (&a, &b)) in actual.iter().zip(&expected.values).enumerate() {
        let error = (f64::from(a) - f64::from(b)).abs();
        ensure!(
            a.is_finite() && error <= 1e-3 + 1e-4 * f64::from(b).abs(),
            "output[{i}]={a}, reference={b}"
        );
        max_error = max_error.max(error);
        squared_error += error * error;
    }
    Ok((max_error, (squared_error / actual.len() as f64).sqrt()))
}

pub fn measure(
    name: &str,
    expected: &Data,
    mut execute: impl FnMut() -> Result<(Vec<usize>, Vec<f32>)>,
) -> Result<Measurement> {
    let start = Instant::now();
    let (shape, output) = execute()?;
    let first_execution_seconds = start.elapsed().as_secs_f64();
    let (max_absolute_error, rmse) =
        validate(&shape, &output, expected).with_context(|| name.to_owned())?;
    drop(output);
    let (warm_shape, warm_output) = execute()?;
    validate(&warm_shape, &warm_output, expected)?;
    drop(warm_output);
    let mut warm_seconds = Vec::new();
    for _ in 0..5 {
        let start = Instant::now();
        let (shape, output) = execute()?;
        warm_seconds.push(start.elapsed().as_secs_f64());
        validate(&shape, &output, expected)?;
    }
    eprintln!("{name}: numerical check passed");
    Ok(Measurement {
        name: name.into(),
        first_execution_seconds,
        warm_seconds,
        max_absolute_error,
        rmse,
        shape,
    })
}
