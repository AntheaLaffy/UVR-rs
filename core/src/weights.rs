//! File identities for reproducible experiments; no checkpoint deserialization.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use md5::Md5;
use sha2::{Digest, Sha256};

// UVR.py:get_model_hash at 5517e0cf0d1acd16a1618eeedec596957523f9e1
// hashes this suffix (or the whole file when shorter) for metadata lookup.
const UVR_HASH_BYTES: u64 = 10_000 * 1024;

#[derive(Debug, PartialEq, Eq)]
pub struct WeightFingerprint {
    pub size_bytes: u64,
    pub sha256: String,
    /// Legacy metadata lookup key, not a whole-file integrity guarantee.
    pub uvr_md5: String,
}

/// Hashes a regular file with bounded memory. Does not validate model contents,
/// provenance, architecture, or compatibility. Do not modify the file during inspection.
pub fn fingerprint(path: &Path) -> io::Result<WeightFingerprint> {
    if !path.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected a regular file",
        ));
    }
    let mut file = File::open(path)?;
    let metadata = file.metadata()?;
    let result = fingerprint_reader(&mut file, metadata.len())?;
    let after = file.metadata()?;
    if after.len() != metadata.len() || after.modified()? != metadata.modified()? {
        return Err(io::Error::other("file changed during inspection"));
    }
    Ok(result)
}

fn fingerprint_reader(reader: &mut impl Read, size_bytes: u64) -> io::Result<WeightFingerprint> {
    let suffix_start = size_bytes.saturating_sub(UVR_HASH_BYTES);
    let mut sha256 = Sha256::new();
    let mut md5 = Md5::new();
    let mut offset = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = match reader.read(&mut buffer) {
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if count == 0 {
            break;
        }
        sha256.update(&buffer[..count]);
        let skip = suffix_start.saturating_sub(offset).min(count as u64) as usize;
        md5.update(&buffer[skip..count]);
        offset += count as u64;
    }
    if offset != size_bytes {
        return Err(io::Error::other("file size changed during inspection"));
    }
    Ok(WeightFingerprint {
        size_bytes,
        sha256: format!("{:x}", sha256.finalize()),
        uvr_md5: format!("{:x}", md5.finalize()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn standard_digest_vectors() {
        for (input, sha256, md5) in [
            (
                "",
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "d41d8cd98f00b204e9800998ecf8427e",
            ),
            (
                "abc",
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                "900150983cd24fb0d6963f7d28e17f72",
            ),
        ] {
            let result = fingerprint_reader(&mut input.as_bytes(), input.len() as u64).unwrap();
            assert_eq!(result.sha256, sha256);
            assert_eq!(result.uvr_md5, md5);
            assert_eq!(result.size_bytes, input.len() as u64);
        }
    }

    #[test]
    fn suffix_boundary_matches_reference() {
        // Independent golden hashes for zero-filled buffers from Node crypto.
        for (size, expected) in [
            (UVR_HASH_BYTES - 1, "a396ed96e4489780d7c683fad6ac7628"),
            (UVR_HASH_BYTES, "596c35b949baf46b721744a13f76a258"),
            (UVR_HASH_BYTES + 1, "596c35b949baf46b721744a13f76a258"),
        ] {
            let data = vec![0; size as usize];
            let result = fingerprint_reader(&mut Cursor::new(data), size).unwrap();
            assert_eq!(result.uvr_md5, expected);
        }
        let mut data = vec![0; UVR_HASH_BYTES as usize + 17];
        let before = fingerprint_reader(&mut Cursor::new(&data), data.len() as u64).unwrap();
        data[..17].fill(1);
        let after = fingerprint_reader(&mut Cursor::new(&data), data.len() as u64).unwrap();
        assert_eq!(before.uvr_md5, after.uvr_md5);
        assert_ne!(before.sha256, after.sha256);
        data[17] = 1;
        let changed = fingerprint_reader(&mut Cursor::new(&data), data.len() as u64).unwrap();
        assert_ne!(after.uvr_md5, changed.uvr_md5);
    }

    #[test]
    fn rejects_changed_length_and_propagates_read_errors() {
        assert!(fingerprint_reader(&mut &b"abc"[..], 4).is_err());
        assert!(fingerprint_reader(&mut &b"abc"[..], 2).is_err());
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::ErrorKind::PermissionDenied.into())
            }
        }
        assert_eq!(
            fingerprint_reader(&mut Broken, 0).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
    }
}
