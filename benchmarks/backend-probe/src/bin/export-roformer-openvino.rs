//! Build a complete FP32 spectral graph directly from the original checkpoint.

use std::path::Path;

use anyhow::{Result, ensure};

#[path = "../roformer_ir.rs"]
mod roformer_ir;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 3 || args.len() == 4,
        "usage: export-roformer-openvino <original.ckpt> <frames:4..801> <network.xml> [batch:1..8]"
    );
    roformer_ir::export(
        Path::new(&args[0]),
        args[1].parse()?,
        Path::new(&args[2]),
        args.get(3).map_or(Ok(1), |value| value.parse())?,
    )
}
