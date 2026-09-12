use std::{collections::BTreeSet, path::Path};

use anyhow::{Context, Result, ensure};
use burn_flex::Flex;
use burn_store::pytorch::PytorchReader;
use burn_tensor::{DType, Tensor, TensorData};

/// Every expected tensor is consumed exactly once; finish rejects unknown keys.
pub(crate) struct Loader {
    reader: PytorchReader,
    used: BTreeSet<String>,
}

impl Loader {
    pub(crate) fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            reader: PytorchReader::new(path)?,
            used: BTreeSet::new(),
        })
    }

    pub(crate) fn finish(self, expected: usize) -> Result<()> {
        ensure!(
            self.reader.len() == expected && self.used.len() == self.reader.len(),
            "checkpoint contains missing or unused tensor keys"
        );
        Ok(())
    }

    pub(crate) fn data(&mut self, name: &str, shape: &[usize], dtype: DType) -> Result<TensorData> {
        ensure!(
            self.used.insert(name.into()),
            "duplicate tensor mapping: {name}"
        );
        let tensor = self
            .reader
            .get(name)
            .with_context(|| format!("missing tensor: {name}"))?;
        ensure!(
            tensor.shape.as_slice() == shape && tensor.dtype == dtype,
            "unexpected shape or dtype: {name}"
        );
        tensor
            .to_data()
            .with_context(|| format!("cannot read tensor: {name}"))
    }

    pub(crate) fn float(&mut self, name: &str, shape: &[usize]) -> Result<TensorData> {
        let data = self.data(name, shape, DType::F32)?;
        let values = data
            .as_slice::<f32>()
            .map_err(|error| anyhow::anyhow!("{name}: {error:?}"))?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite weights: {name}"
        );
        Ok(data)
    }

    pub(crate) fn channel(&mut self, name: &str, channels: usize) -> Result<Tensor<Flex, 4>> {
        let mut data = self.float(name, &[channels])?;
        data.shape = [1, channels, 1, 1].into();
        Ok(Tensor::from_data(data, &Default::default()))
    }
}
