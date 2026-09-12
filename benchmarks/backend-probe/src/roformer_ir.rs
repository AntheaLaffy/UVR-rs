//! Export the shared Rust checkpoint graph for independent backend experiments.

use anyhow::{Result, ensure};
use serde_json::json;
use std::{fs::OpenOptions, io::Write, path::Path, time::Instant};

pub fn export(checkpoint: &Path, frames: usize, output: &Path, batch: usize) -> Result<()> {
    ensure!((1..=8).contains(&batch), "batch must be 1..8");
    ensure!((4..=801).contains(&frames), "frame count must be 4..801");
    ensure!(
        output.extension().is_some_and(|v| v == "xml"),
        "output must have .xml extension"
    );
    ensure!(
        !output.exists()
            && !output.with_extension("bin").exists()
            && !output.with_extension("provenance.json").exists(),
        "output already exists"
    );
    let start = Instant::now();
    let graph =
        uvr_core::roformer::openvino::RoformerIr::from_checkpoint_with_batch(
            checkpoint,
            batch,
            frames,
            || true,
        )?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("bin"))?
        .write_all(&graph.weights)?;
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?
        .write_all(&graph.xml)?;
    let provenance = json!({
        "generator": "Rust original checkpoint -> OpenVINO IR v11; no Python or reference IR input",
        "checkpoint_sha256": graph.checkpoint_sha256, "batch": batch, "frames": frames, "input_shape": [batch, frames, 4100],
        "strict_tensor_count": 699, "operations": graph.operations, "weights_bytes": graph.weights.len(),
        "export_seconds": start.elapsed().as_secs_f64(),
        "xml_sha256": uvr_core::weights::fingerprint(output)?.sha256,
        "bin_sha256": uvr_core::weights::fingerprint(&output.with_extension("bin"))?.sha256,
        "generator_sha256": uvr_core::weights::fingerprint(&std::env::current_exe()?)?.sha256,
        "precision": "FP32 network, exact GELU, L2 clamp 1e-12; DSP remains outside this graph",
        "scope": "experimental graph; requires complete waveform, memory and latency validation per shape"
    });
    let mut report = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output.with_extension("provenance.json"))?;
    serde_json::to_writer_pretty(&mut report, &provenance)?;
    writeln!(report)?;
    eprintln!(
        "Exported {} original tensors into {} operations",
        699, graph.operations
    );
    Ok(())
}
