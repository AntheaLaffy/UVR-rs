use serde::Deserialize;

#[derive(Clone, Copy, Default, Deserialize)]
pub enum Locale {
    #[default]
    #[serde(rename = "zh-CN")]
    Chinese,
    #[serde(rename = "en")]
    English,
    #[serde(rename = "ja")]
    Japanese,
}

impl Locale {
    pub fn text<'a>(self, chinese: &'a str, english: &'a str, japanese: &'a str) -> &'a str {
        match self {
            Self::Chinese => chinese,
            Self::English => english,
            Self::Japanese => japanese,
        }
    }

    pub fn error(self, message: String) -> String {
        match message.as_str() {
            "任务标识不正确" => self.text(
                "任务标识不正确",
                "Invalid task ID",
                "タスクIDが正しくありません",
            ),
            "任务状态不可用" => self.text(
                "任务状态不可用",
                "Task state is unavailable",
                "タスクの状態を取得できません",
            ),
            "已有任务正在处理，请先等待或取消" => self.text(
                "已有任务正在处理，请先等待或取消",
                "Wait for or cancel the current task",
                "実行中のタスクが完了するまで待つか、キャンセルしてください",
            ),
            "1296 不接受 VR 参数" => self.text(
                "1296 不接受 VR 参数",
                "1296 does not accept VR options",
                "1296ではVRの設定を使用できません",
            ),
            "RoFormer 批次与布局参数仅适用于 1296 的 Burn 后端" => self.text(
                "RoFormer 批次与布局参数仅适用于 1296 的 Burn 后端",
                "RoFormer batch and layout settings apply only to Burn 1296",
                "RoFormerのバッチとレイアウト設定はBurn 1296専用です",
            ),
            _ => return message,
        }
        .into()
    }
}
