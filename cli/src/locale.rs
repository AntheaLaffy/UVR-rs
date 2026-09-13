use std::ffi::OsString;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locale {
    ZhCn,
    En,
    Ja,
}

macro_rules! tr {
    ($locale:expr, $zh:literal, $en:literal, $ja:literal $(, $arg:expr)* $(,)?) => {
        match $locale {
            crate::locale::Locale::ZhCn => format!($zh $(, $arg)*),
            crate::locale::Locale::En => format!($en $(, $arg)*),
            crate::locale::Locale::Ja => format!($ja $(, $arg)*),
        }
    };
}

impl Locale {
    pub fn parse(args: &[OsString]) -> Result<(Self, &[OsString]), (Self, String)> {
        let explicit = args.first().is_some_and(|value| value == "--lang");
        let from_environment;
        let value = if explicit {
            args.get(1)
                .filter(|value| !value.to_string_lossy().starts_with('-'))
                .ok_or((Self::ZhCn, "--lang 缺少值；请选择 zh-CN、en 或 ja".into()))?
        } else {
            from_environment = std::env::var_os("UVR_LANG");
            match &from_environment {
                Some(value) => value,
                None => return Ok((Self::ZhCn, args)),
            }
        };
        let locale = match value.to_str() {
            Some("zh-CN") => Self::ZhCn,
            Some("en") => Self::En,
            Some("ja") => Self::Ja,
            _ => {
                let source = if explicit { "--lang" } else { "UVR_LANG" };
                return Err((Self::ZhCn, format!("{source} 必须为 zh-CN、en 或 ja")));
            }
        };
        let args = if explicit { &args[2..] } else { args };
        if args.first().is_some_and(|value| value == "--lang") {
            return Err((
                locale,
                tr!(
                    locale,
                    "选项重复：--lang",
                    "Duplicate option: --lang",
                    "オプションが重複しています：--lang"
                ),
            ));
        }
        Ok((locale, args))
    }

    pub fn help(self) -> &'static str {
        match self {
            Self::ZhCn => include_str!("help.zh-CN.txt"),
            Self::En => include_str!("help.en.txt"),
            Self::Ja => include_str!("help.ja.txt"),
        }
    }
}
