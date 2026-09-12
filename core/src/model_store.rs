//! External model discovery and verified downloads, independent of audio inference.

use std::{
    fs::{self, File},
    future::Future,
    io::{Read, Write},
    ops::ControlFlow,
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use reqwest::{Client, Proxy, StatusCode};
use sha2::{Digest, Sha256};

use crate::{model_catalog::ModelInfo, task::TaskCancelled};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    Missing,
    Ready,
    Invalid(String),
}

#[derive(Debug, Clone, Copy, Default)]
pub enum DownloadSource {
    #[default]
    HuggingFace,
    GitHub,
}

#[derive(Debug, Default)]
pub struct DownloadOptions {
    pub source: DownloadSource,
    /// None uses the environment/system proxy configuration.
    pub proxy: Option<String>,
    /// Explicit repair action. Valid weights are always kept.
    pub replace_invalid: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadStage {
    Checking,
    Connecting,
    Receiving,
    Verifying,
    Complete,
}

#[derive(Debug, Clone, Copy)]
pub struct DownloadProgress {
    pub stage: DownloadStage,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug)]
pub struct DownloadOutput {
    pub path: PathBuf,
    pub downloaded: bool,
}

fn check(flow: ControlFlow<()>) -> Result<()> {
    if flow.is_break() {
        Err(TaskCancelled.into())
    } else {
        Ok(())
    }
}

fn model_path(info: ModelInfo, directory: &Path) -> Result<PathBuf> {
    let mut components = Path::new(info.filename).components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "model filename must be a single normal path component"
    );
    ensure!(!directory.as_os_str().is_empty(), "请选择模型目录");
    Ok(directory.join(info.filename))
}

/// Follows a directory or file symlink. A name or matching size alone is never Ready.
pub fn inspect(
    info: ModelInfo,
    directory: &Path,
    mut progress: impl FnMut(u64) -> ControlFlow<()>,
) -> Result<Availability> {
    let path = model_path(info, directory)?;
    check(progress(0))?;
    let metadata = match path.metadata() {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(if path.symlink_metadata().is_ok() {
                Availability::Invalid("模型文件软链接的目标不存在".into())
            } else {
                Availability::Missing
            });
        }
        Err(error) => {
            return Err(error).with_context(|| format!("无法读取模型：{}", path.display()));
        }
    };
    if !metadata.is_file() {
        return Ok(Availability::Invalid("模型路径不是普通文件".into()));
    }
    if metadata.len() != info.size_bytes {
        return Ok(Availability::Invalid(format!(
            "文件大小不符：{} / {} 字节",
            metadata.len(),
            info.size_bytes
        )));
    }
    let mut file = File::open(&path).context("无法打开模型文件")?;
    let before = file.metadata()?;
    let mut sha = Sha256::new();
    let mut count = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        check(progress(count))?;
        let length = match file.read(&mut buffer) {
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            result => result?,
        };
        if length == 0 {
            break;
        }
        count += length as u64;
        sha.update(&buffer[..length]);
    }
    let after = file.metadata()?;
    ensure!(
        before.len() == after.len()
            && before.modified()? == after.modified()?
            && count == info.size_bytes,
        "校验期间模型文件发生变化，请重新检测"
    );
    check(progress(count))?;
    Ok(if format!("{:x}", sha.finalize()) == info.sha256 {
        Availability::Ready
    } else {
        Availability::Invalid("SHA-256 不符，文件损坏或不是指定版本".into())
    })
}

fn client(options: &DownloadOptions) -> Result<Client> {
    let mut builder = Client::builder()
        .user_agent(concat!("uvr-rust/", env!("CARGO_PKG_VERSION")))
        .https_only(true)
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(30));
    if let Some(proxy) = options
        .proxy
        .as_deref()
        .map(str::trim)
        .filter(|p| !p.is_empty())
    {
        let url = reqwest::Url::parse(proxy).map_err(|_| anyhow::anyhow!("代理地址格式不正确"))?;
        ensure!(
            matches!(url.scheme(), "http" | "https" | "socks5" | "socks5h"),
            "代理需使用 http、https、socks5 或 socks5h 地址"
        );
        builder =
            builder.proxy(Proxy::all(url).map_err(|_| anyhow::anyhow!("无法使用此代理地址"))?);
    }
    builder
        .build()
        .map_err(|error| error.without_url())
        .context("无法创建下载连接")
}

/// A stalled connection remains cancellable. Dropping the request also stops its body.
async fn receive<T>(
    future: impl Future<Output = reqwest::Result<T>>,
    current: DownloadProgress,
    progress: &mut impl FnMut(DownloadProgress) -> ControlFlow<()>,
) -> Result<T> {
    tokio::pin!(future);
    let mut tick = tokio::time::interval(Duration::from_millis(100));
    loop {
        tokio::select! {
            result = &mut future => return result.map_err(|error| error.without_url())
                .context("下载连接失败，请检查网络或代理"),
            _ = tick.tick() => check(progress(current))?,
        }
    }
}

/// Downloads a pinned source to a temporary file in the selected directory.
/// Only complete, size-checked and SHA-256-verified bytes become a loadable filename.
pub async fn download(
    info: ModelInfo,
    directory: &Path,
    options: &DownloadOptions,
    progress: impl FnMut(DownloadProgress) -> ControlFlow<()>,
) -> Result<DownloadOutput> {
    let url = match options.source {
        DownloadSource::HuggingFace => info.hugging_face_url(),
        DownloadSource::GitHub => info.github_url(),
    };
    download_from(info, directory, options, || client(options), &url, progress).await
}

async fn download_from(
    info: ModelInfo,
    directory: &Path,
    options: &DownloadOptions,
    make_client: impl FnOnce() -> Result<Client>,
    url: &str,
    mut progress: impl FnMut(DownloadProgress) -> ControlFlow<()>,
) -> Result<DownloadOutput> {
    let path = model_path(info, directory)?;
    let mut current = DownloadProgress {
        stage: DownloadStage::Checking,
        completed_bytes: 0,
        total_bytes: info.size_bytes,
    };
    let existing = inspect(info, directory, |bytes| {
        progress(DownloadProgress {
            completed_bytes: bytes,
            ..current
        })
    })?;
    if existing == Availability::Ready {
        let _ = progress(DownloadProgress {
            stage: DownloadStage::Complete,
            completed_bytes: info.size_bytes,
            ..current
        });
        return Ok(DownloadOutput {
            path,
            downloaded: false,
        });
    }
    if let Availability::Invalid(reason) = existing {
        ensure!(
            options.replace_invalid,
            "{reason}；请选择重新下载或其他目录"
        );
        ensure!(
            path.symlink_metadata()?.file_type().is_file(),
            "不能替换目录或模型文件软链接，请修复链接目标或选择其他目录"
        );
    }
    check(progress(current))?;
    fs::create_dir_all(directory)
        .context("无法创建模型目录，请选择可写目录（目录软链接需有有效目标）")?;
    let mut temporary = tempfile::Builder::new()
        .prefix(".uvr-download-")
        .suffix(".part")
        .tempfile_in(directory)
        .context("模型目录不可写，请选择其他目录")?;
    let client = make_client()?;
    current.stage = DownloadStage::Connecting;
    check(progress(current))?;
    let mut response = receive(
        client
            .get(url)
            .header(reqwest::header::ACCEPT_ENCODING, "identity")
            .send(),
        current,
        &mut progress,
    )
    .await?;
    ensure!(
        response.status() == StatusCode::OK,
        "下载服务器返回 HTTP {}",
        response.status()
    );
    if let Some(length) = response.content_length() {
        ensure!(
            length == info.size_bytes,
            "下载内容长度不符：{length} / {} 字节",
            info.size_bytes
        );
    }
    current.stage = DownloadStage::Receiving;
    let mut sha = Sha256::new();
    loop {
        check(progress(current))?;
        let Some(chunk) = receive(response.chunk(), current, &mut progress).await? else {
            break;
        };
        ensure!(
            chunk.len() as u64 <= info.size_bytes.saturating_sub(current.completed_bytes),
            "下载内容超过指定模型大小"
        );
        temporary
            .write_all(&chunk)
            .context("无法保存模型，请检查空间与目录权限")?;
        sha.update(&chunk);
        current.completed_bytes += chunk.len() as u64;
    }
    current.stage = DownloadStage::Verifying;
    check(progress(current))?;
    ensure!(
        current.completed_bytes == info.size_bytes,
        "下载未完成：{} / {} 字节",
        current.completed_bytes,
        info.size_bytes
    );
    ensure!(
        format!("{:x}", sha.finalize()) == info.sha256,
        "下载文件 SHA-256 校验失败，未替换现有模型"
    );
    temporary.as_file().sync_all().context("无法完成模型写入")?;
    // Another instance may have installed the same model while the request ran.
    let now = inspect(info, directory, |_| progress(current))?;
    check(progress(current))?;
    match now {
        Availability::Ready => {
            let _ = progress(DownloadProgress {
                stage: DownloadStage::Complete,
                ..current
            });
            return Ok(DownloadOutput {
                path,
                downloaded: false,
            });
        }
        Availability::Missing => {
            temporary
                .persist_noclobber(&path)
                .context("模型文件已出现，未覆盖；请重新检测")?;
        }
        Availability::Invalid(_) if options.replace_invalid => {
            ensure!(
                path.symlink_metadata()?.file_type().is_file(),
                "模型路径已变成软链接或目录，未替换"
            );
            temporary.persist(&path).context("无法替换损坏的模型文件")?;
        }
        Availability::Invalid(_) => bail!("下载期间模型路径已被占用，未覆盖；请重新检测"),
    }
    let _ = progress(DownloadProgress {
        stage: DownloadStage::Complete,
        ..current
    });
    Ok(DownloadOutput {
        path,
        downloaded: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{net::TcpListener, thread, time::Instant};

    const SAMPLE: ModelInfo = ModelInfo {
        key: "sample",
        label: "sample",
        filename: "model.bin",
        size_bytes: 3,
        sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    };

    fn http_response(response: Vec<u8>, stall: bool) -> (String, thread::JoinHandle<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model.bin", listener.local_addr().unwrap());
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8];
            while !request.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            socket.write_all(&response).unwrap();
            if stall {
                let _ = socket.read(&mut byte);
            }
            request
        });
        (url, server)
    }

    fn test_client() -> Result<Client> {
        Ok(Client::builder()
            .no_proxy()
            .read_timeout(Duration::from_secs(3))
            .build()?)
    }

    #[tokio::test]
    async fn explicit_proxy_handles_https_without_exposing_credentials() {
        let directory = tempfile::tempdir().unwrap();
        let (address, server) = http_response(
            b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n".to_vec(),
            false,
        );
        let address = reqwest::Url::parse(&address).unwrap();
        let options = DownloadOptions {
            proxy: Some(format!(
                "http://download-user:download-secret@127.0.0.1:{}",
                address.port().unwrap()
            )),
            ..Default::default()
        };
        let error = download_from(
            SAMPLE,
            directory.path(),
            &options,
            || client(&options),
            "https://example.invalid/model.bin",
            |_| ControlFlow::Continue(()),
        )
        .await
        .unwrap_err();
        let request = server.join().unwrap();
        assert!(request.starts_with(b"CONNECT example.invalid:443 HTTP/1.1\r\n"));
        let message = format!("{error:#}");
        assert!(!message.contains("download-user") && !message.contains("download-secret"));
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    }

    #[test]
    fn availability_requires_content_and_supports_cancellation() {
        let directory = tempfile::tempdir().unwrap();
        assert_eq!(
            inspect(SAMPLE, directory.path(), |_| ControlFlow::Continue(())).unwrap(),
            Availability::Missing
        );
        let path = directory.path().join(SAMPLE.filename);
        for bytes in [b"".as_slice(), b"abd", b"abcd"] {
            fs::write(&path, bytes).unwrap();
            assert!(matches!(
                inspect(SAMPLE, directory.path(), |_| ControlFlow::Continue(())).unwrap(),
                Availability::Invalid(_)
            ));
        }
        fs::write(&path, b"abc").unwrap();
        assert_eq!(
            inspect(SAMPLE, directory.path(), |_| ControlFlow::Continue(())).unwrap(),
            Availability::Ready
        );
        assert!(
            inspect(SAMPLE, directory.path(), |_| ControlFlow::Break(()))
                .unwrap_err()
                .is::<TaskCancelled>()
        );
        let bad = ModelInfo {
            filename: "../outside",
            ..SAMPLE
        };
        assert!(inspect(bad, directory.path(), |_| ControlFlow::Continue(())).is_err());
    }

    #[tokio::test]
    async fn verifies_download_and_skips_ready_file_without_a_connection() {
        let directory = tempfile::tempdir().unwrap();
        let (url, server) = http_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc".to_vec(),
            false,
        );
        let options = DownloadOptions::default();
        let result = download_from(
            SAMPLE,
            directory.path(),
            &options,
            test_client,
            &url,
            |_| ControlFlow::Continue(()),
        )
        .await
        .unwrap();
        server.join().unwrap();
        assert!(result.downloaded);
        assert_eq!(fs::read(&result.path).unwrap(), b"abc");
        let skipped = download_from(
            SAMPLE,
            directory.path(),
            &options,
            || panic!("ready model must not connect"),
            &url,
            |_| ControlFlow::Continue(()),
        )
        .await
        .unwrap();
        assert!(!skipped.downloaded);
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[tokio::test]
    async fn rejects_http_error_truncation_and_wrong_hash_without_publishing() {
        for response in [
            b"HTTP/1.1 404 Not Found\r\nContent-Length: 3\r\n\r\nabc".as_slice(),
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nab",
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabd",
            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nabcd",
        ] {
            let directory = tempfile::tempdir().unwrap();
            let (url, server) = http_response(response.to_vec(), false);
            assert!(
                download_from(
                    SAMPLE,
                    directory.path(),
                    &DownloadOptions::default(),
                    test_client,
                    &url,
                    |_| ControlFlow::Continue(())
                )
                .await
                .is_err()
            );
            server.join().unwrap();
            assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
        }
    }

    #[tokio::test]
    async fn stalled_body_cancels_promptly_and_preserves_existing_file() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(SAMPLE.filename);
        fs::write(&path, b"bad").unwrap();
        let (url, server) = http_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n".to_vec(),
            true,
        );
        let options = DownloadOptions {
            replace_invalid: true,
            ..Default::default()
        };
        let start = Instant::now();
        let error = download_from(
            SAMPLE,
            directory.path(),
            &options,
            test_client,
            &url,
            |_| {
                if start.elapsed() >= Duration::from_millis(150) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
        .await
        .unwrap_err();
        assert!(error.is::<TaskCancelled>());
        assert!(start.elapsed() < Duration::from_secs(2));
        server.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"bad");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[tokio::test]
    async fn repairs_invalid_file_only_when_requested() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(SAMPLE.filename);
        fs::write(&path, b"bad").unwrap();
        assert!(
            download_from(
                SAMPLE,
                directory.path(),
                &DownloadOptions::default(),
                || panic!("repair requires an explicit option"),
                "unused",
                |_| ControlFlow::Continue(())
            )
            .await
            .is_err()
        );
        let (url, server) = http_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc".to_vec(),
            false,
        );
        let options = DownloadOptions {
            replace_invalid: true,
            ..Default::default()
        };
        download_from(
            SAMPLE,
            directory.path(),
            &options,
            test_client,
            &url,
            |_| ControlFlow::Continue(()),
        )
        .await
        .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"abc");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn directory_symlink_and_portable_discovery_work() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let storage = root.path().join("actual storage");
        let portable = root.path().join("portable");
        let cwd = root.path().join("cwd");
        fs::create_dir_all(&storage).unwrap();
        fs::create_dir_all(&portable).unwrap();
        let linked = portable.join("models");
        symlink(&storage, &linked).unwrap();
        assert_eq!(
            crate::model_catalog::default_directory(&portable.join("uvr-gui"), &cwd),
            linked
        );
        let (url, server) = http_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc".to_vec(),
            false,
        );
        download_from(
            SAMPLE,
            &linked,
            &DownloadOptions::default(),
            test_client,
            &url,
            |_| ControlFlow::Continue(()),
        )
        .await
        .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(storage.join(SAMPLE.filename)).unwrap(), b"abc");
        assert!(linked.symlink_metadata().unwrap().file_type().is_symlink());
        let aliased = ModelInfo {
            filename: "alias.bin",
            ..SAMPLE
        };
        symlink(storage.join(SAMPLE.filename), linked.join(aliased.filename)).unwrap();
        assert_eq!(
            inspect(aliased, &linked, |_| ControlFlow::Continue(())).unwrap(),
            Availability::Ready
        );
        assert_eq!(
            crate::model_catalog::default_directory(&root.path().join("light/uvr-gui"), &cwd),
            root.path().join("light/models")
        );
    }
}
