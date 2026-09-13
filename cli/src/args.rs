use std::{collections::BTreeMap, ffi::OsString, path::PathBuf};

use crate::locale::Locale;

use uvr_core::{
    file_task::ModelSpec,
    runtime::{LinearLayout, RuntimeBackend, RuntimeOptions},
};

#[derive(Debug)]
pub struct Separation {
    pub spec: ModelSpec,
    pub weights: PathBuf,
    pub input: PathBuf,
    pub directory: PathBuf,
    pub runtime: RuntimeOptions,
}

pub fn separation(args: &[OsString], locale: Locale) -> Result<Separation, String> {
    separation_with_env(args, locale, |name| std::env::var_os(name))
}

fn separation_with_env(
    args: &[OsString],
    locale: Locale,
    environment: impl Fn(&str) -> Option<OsString>,
) -> Result<Separation, String> {
    let vr = args.first().is_some_and(|command| command == "separate-vr");
    let positional = if vr { 5 } else { 4 };
    if args.len() < positional {
        return Err(tr!(
            locale,
            "缺少模型、权重、音频或输出目录参数",
            "Missing model, weights, audio or output directory arguments",
            "モデル、重み、音声、または出力先の引数が不足しています"
        ));
    }
    let spec = if vr {
        let model_error = tr!(
            locale,
            "VR 模型必须为 5hp、6hp 或 deecho",
            "VR model must be 5hp, 6hp or deecho",
            "VR モデルは 5hp、6hp、deecho のいずれかです"
        );
        let key = args[1].to_str().ok_or_else(|| model_error.clone())?;
        ModelSpec::from_key(key)
            .filter(|spec| matches!(spec, ModelSpec::Vr { .. }))
            .ok_or(model_error)?
    } else {
        ModelSpec::Roformer1296
    };
    let mut supplied = BTreeMap::new();
    let mut remaining = args[positional..].iter();
    while let Some(option) = remaining.next() {
        let option = option.to_str().ok_or_else(|| {
            tr!(
                locale,
                "选项名称必须是有效 UTF-8",
                "Option names must be valid UTF-8",
                "オプション名は有効な UTF-8 にしてください"
            )
        })?;
        match option {
            "--threads" | "--parallel-windows" => (),
            "--window-frames" | "--inference-batch" if vr => (),
            "--backend" | "--time-batch" | "--frequency-batch" | "--linear-layout" if !vr => (),
            "--window-frames" | "--inference-batch" | "--backend" | "--time-batch"
            | "--frequency-batch" | "--linear-layout" => {
                return Err(tr!(
                    locale,
                    "{option} 不适用于模型 {}",
                    "{option} does not apply to model {}",
                    "{option} はモデル {} には適用できません",
                    spec.key()
                ));
            }
            _ => {
                return Err(tr!(
                    locale,
                    "未知选项：{option}",
                    "Unknown option: {option}",
                    "不明なオプション：{option}"
                ));
            }
        }
        if supplied.contains_key(option) {
            return Err(tr!(
                locale,
                "选项重复：{option}",
                "Duplicate option: {option}",
                "オプションが重複しています：{option}"
            ));
        }
        let value = remaining
            .next()
            .filter(|value| !value.to_string_lossy().starts_with("--"))
            .ok_or_else(|| {
                tr!(
                    locale,
                    "{option} 缺少值",
                    "Missing value for {option}",
                    "{option} の値がありません"
                )
            })?;
        supplied.insert(option, value);
    }
    let mut runtime = RuntimeOptions::for_model(spec);
    if let Some(value) = supplied.get("--backend") {
        runtime.backend = match value.to_str() {
            Some("burn") => RuntimeBackend::Burn,
            Some("openvino-cpu") => RuntimeBackend::OpenvinoCpu,
            _ => {
                return Err(tr!(
                    locale,
                    "--backend 必须为 burn 或 openvino-cpu",
                    "--backend must be burn or openvino-cpu",
                    "--backend は burn または openvino-cpu です"
                ));
            }
        };
    }
    if runtime.backend == RuntimeBackend::OpenvinoCpu {
        for option in [
            "--time-batch",
            "--frequency-batch",
            "--parallel-windows",
            "--linear-layout",
        ] {
            if supplied.contains_key(option) {
                return Err(tr!(
                    locale,
                    "{option} 仅适用于 Burn 后端",
                    "{option} applies only to the Burn backend",
                    "{option} は Burn バックエンド専用です"
                ));
            }
        }
    }
    let number = |option: &str, variable: Option<&str>, default: usize| {
        if let Some(value) = supplied.get(option) {
            return positive_integer(option, value, locale);
        }
        if let Some(variable) = variable
            && let Some(value) = environment(variable)
        {
            return positive_integer(variable, &value, locale);
        }
        Ok(default)
    };
    runtime.threads = number("--threads", Some("RAYON_NUM_THREADS"), runtime.threads)?;
    if vr {
        runtime.vr.window_frames = number("--window-frames", None, runtime.vr.window_frames)?;
        runtime.vr.inference_batch = number("--inference-batch", None, runtime.vr.inference_batch)?;
        runtime.vr.window_parallelism =
            number("--parallel-windows", None, runtime.vr.window_parallelism)?;
        if runtime.vr.inference_batch > 4 {
            return Err(tr!(
                locale,
                "--inference-batch 必须在 1 到 4 之间",
                "--inference-batch must be between 1 and 4",
                "--inference-batch は 1～4 の範囲で指定してください"
            ));
        }
        if runtime.vr.window_parallelism > 8 {
            return Err(tr!(
                locale,
                "--parallel-windows 必须在 1 到 8 之间",
                "--parallel-windows must be between 1 and 8",
                "--parallel-windows は 1～8 の範囲で指定してください"
            ));
        }
    } else if runtime.backend == RuntimeBackend::Burn {
        runtime.roformer.time_batch = number(
            "--time-batch",
            Some("UVR_ROFORMER_TIME_BATCH"),
            runtime.roformer.time_batch,
        )?;
        runtime.roformer.frequency_batch = number(
            "--frequency-batch",
            Some("UVR_ROFORMER_FREQUENCY_BATCH"),
            runtime.roformer.frequency_batch,
        )?;
        runtime.roformer.window_parallelism = number(
            "--parallel-windows",
            Some("UVR_ROFORMER_WINDOW_PARALLELISM"),
            runtime.roformer.window_parallelism,
        )?;
        let layout = supplied
            .get("--linear-layout")
            .map(|value| ("--linear-layout", (*value).clone()))
            .or_else(|| environment("UVR_LINEAR_LAYOUT").map(|value| ("UVR_LINEAR_LAYOUT", value)));
        if let Some((source, value)) = layout {
            runtime.roformer.linear_layout = match value.to_str() {
                Some("flattened") => LinearLayout::Flattened,
                Some("batched") => LinearLayout::Batched,
                _ => {
                    return Err(tr!(
                        locale,
                        "{source} 必须为 flattened 或 batched",
                        "{source} must be flattened or batched",
                        "{source} は flattened または batched です"
                    ));
                }
            };
        }
    }
    let runtime = runtime.effective_for(spec).map_err(|error| {
        tr!(
            locale,
            "运行时参数无效：{error}",
            "Invalid runtime settings: {error}",
            "推論設定が無効です：{error}"
        )
    })?;
    Ok(Separation {
        spec,
        weights: PathBuf::from(&args[positional - 3]),
        input: PathBuf::from(&args[positional - 2]),
        directory: PathBuf::from(&args[positional - 1]),
        runtime,
    })
}

fn positive_integer(source: &str, value: &OsString, locale: Locale) -> Result<usize, String> {
    let parsed = value.to_str().and_then(|value| {
        (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
            .then(|| value.parse::<usize>().ok())
            .flatten()
    });
    parsed.filter(|&value| value > 0).ok_or_else(|| {
        tr!(
            locale,
            "{source} 必须为可表示的正整数",
            "{source} must be a representable positive integer",
            "{source} は表現可能な正の整数で指定してください"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str], environment: &[(&str, &str)]) -> Result<Separation, String> {
        let args: Vec<_> = args.iter().map(OsString::from).collect();
        separation_with_env(&args, Locale::ZhCn, |key| {
            environment
                .iter()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| OsString::from(value))
        })
    }

    #[test]
    fn explicit_settings_override_environment_per_field() {
        let command = parse(
            &[
                "separate-1296",
                "weights with spaces",
                "input",
                "output",
                "--backend",
                "burn",
                "--threads",
                "3",
                "--time-batch",
                "8",
                "--linear-layout",
                "batched",
            ],
            &[
                ("RAYON_NUM_THREADS", "bad"),
                ("UVR_ROFORMER_TIME_BATCH", "bad"),
                ("UVR_ROFORMER_FREQUENCY_BATCH", "128"),
                ("UVR_ROFORMER_WINDOW_PARALLELISM", "2"),
                ("UVR_LINEAR_LAYOUT", "bad"),
            ],
        )
        .unwrap();
        assert_eq!(command.weights, PathBuf::from("weights with spaces"));
        assert_eq!(command.runtime.threads, 3);
        assert_eq!(command.runtime.roformer.time_batch, 8);
        assert_eq!(command.runtime.roformer.frequency_batch, 128);
        assert_eq!(command.runtime.roformer.window_parallelism, 2);
        assert_eq!(
            command.runtime.roformer.linear_layout,
            LinearLayout::Batched
        );
    }

    #[test]
    fn vr_ignores_roformer_environment_and_reports_effective_options() {
        let environment = &[
            ("UVR_ROFORMER_TIME_BATCH", "bad"),
            ("UVR_LINEAR_LAYOUT", "bad"),
        ];
        let hp = parse(
            &[
                "separate-vr",
                "5hp",
                "w",
                "i",
                "o",
                "--inference-batch",
                "2",
                "--parallel-windows",
                "4",
            ],
            environment,
        )
        .unwrap();
        assert_eq!(hp.runtime.vr.inference_batch, 2);
        assert_eq!(hp.runtime.vr.window_parallelism, 1);
        let deecho = parse(
            &[
                "separate-vr",
                "deecho",
                "w",
                "i",
                "o",
                "--inference-batch",
                "4",
                "--parallel-windows",
                "8",
            ],
            environment,
        )
        .unwrap();
        assert_eq!(deecho.runtime.vr.inference_batch, 1);
        assert_eq!(deecho.runtime.vr.window_parallelism, 1);
    }

    #[test]
    fn rejects_ambiguous_and_invalid_options_before_opening_files() {
        for (options, expected) in [
            (vec!["--threads"], "--threads 缺少值"),
            (vec!["--threads", "--time-batch", "2"], "--threads 缺少值"),
            (vec!["--threads", "2", "--threads", "3"], "选项重复"),
            (vec!["--unknown", "2"], "未知选项"),
            (vec!["--threads", "0"], "--threads"),
            (vec!["--threads", "-2"], "--threads"),
            (vec!["--threads", "+2"], "--threads"),
            (vec!["--time-batch", "1.5"], "--time-batch"),
            (
                vec!["--frequency-batch", "9999999999999999999999999999"],
                "--frequency-batch",
            ),
            (vec!["--parallel-windows", "9"], "window parallelism"),
            (vec!["--linear-layout", "invalid"], "--linear-layout"),
            (vec!["--backend", "cuda"], "--backend"),
            (vec!["--window-frames", "512"], "不适用于"),
            (vec!["--inference-batch", "1"], "不适用于"),
            (
                vec!["--backend", "openvino-cpu", "--time-batch", "1"],
                "仅适用于 Burn",
            ),
        ] {
            let mut args = vec!["separate-1296", "w", "i", "o"];
            if !options.contains(&"--backend") {
                args.extend(["--backend", "burn"]);
            }
            args.extend(options);
            let error = parse(&args, &[]).unwrap_err();
            assert!(error.contains(expected), "{args:?}: {error}");
        }
    }

    #[test]
    fn invalid_active_environment_is_an_argument_error() {
        for variable in [
            "RAYON_NUM_THREADS",
            "UVR_ROFORMER_TIME_BATCH",
            "UVR_ROFORMER_FREQUENCY_BATCH",
            "UVR_ROFORMER_WINDOW_PARALLELISM",
            "UVR_LINEAR_LAYOUT",
        ] {
            let error = parse(
                &["separate-1296", "w", "i", "o", "--backend", "burn"],
                &[(variable, "bad")],
            )
            .unwrap_err();
            assert!(error.contains(variable), "{error}");
        }
    }

    #[cfg(feature = "openvino")]
    #[test]
    fn openvino_ignores_burn_environment() {
        let result = parse(
            &["separate-1296", "w", "i", "o", "--backend", "openvino-cpu"],
            &[
                ("UVR_ROFORMER_TIME_BATCH", "bad"),
                ("UVR_ROFORMER_FREQUENCY_BATCH", "bad"),
                ("UVR_ROFORMER_WINDOW_PARALLELISM", "bad"),
                ("UVR_LINEAR_LAYOUT", "bad"),
            ],
        );
        if uvr_core::runtime::openvino_available() {
            assert_eq!(result.unwrap().runtime.backend, RuntimeBackend::OpenvinoCpu);
        } else {
            assert!(
                result
                    .unwrap_err()
                    .contains("OpenVINO CPU runtime is unavailable")
            );
        }
    }
}
