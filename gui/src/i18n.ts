export type Language = "zh-CN" | "ja" | "en";
export type MessageArgs = Record<string, string | number>;

// Keep every user-facing translation together so dynamic and static UI agree.
const messages: Record<string, readonly [string, string, string]> = {
  "app.title": ["UVR · 音轨工作台", "UVR · オーディオワークスペース", "UVR · Audio workspace"],
  "app.workspace": ["音轨工作台", "オーディオワークスペース", "Audio workspace"],
  "app.subtitle": ["人声、伴奏与混响分离", "ボーカル・伴奏・残響を分離", "Separate vocals, instrumentals and reverb"],
  "app.badge": ["本地 CPU · 验证版", "ローカル CPU · プレビュー", "Local CPU · Preview"],
  "app.preview": ["网页预览：请在桌面应用中处理本地音频。", "Web プレビューです。ローカル音声の処理にはデスクトップアプリをお使いください。", "Web preview: use the desktop app to process local audio."],
  "version.check": ["检查更新", "更新を確認", "Check for updates"],
  "version.checking": ["正在检查更新…", "更新を確認中…", "Checking for updates…"],
  "version.current": ["当前版本 v{version}", "現在のバージョン v{version}", "Current version v{version}"],
  "version.latest": ["发现新版本 v{version}", "新しいバージョン v{version}があります", "Update available: v{version}"],
  "version.upToDate": ["已是最新版本", "最新バージョンです", "You are up to date"],
  "version.unavailable": ["暂时无法检查更新", "更新を確認できません", "Updates unavailable"],
  "version.release": ["查看 Release", "Release を表示", "View release"],
  "app.language": ["语言", "言語", "Language"],
  "appearance.toggle": ["外观", "外観", "Appearance"],
  "appearance.mode": ["明暗模式", "表示モード", "Color mode"],
  "appearance.system": ["跟随系统", "システムに合わせる", "System"],
  "appearance.light": ["浅色", "ライト", "Light"],
  "appearance.dark": ["深色", "ダーク", "Dark"],
  "appearance.theme": ["主题色", "テーマカラー", "Theme color"],
  "appearance.violet": ["紫罗兰", "バイオレット", "Violet"],
  "appearance.blue": ["海蓝", "ブルー", "Blue"],
  "appearance.teal": ["青绿", "ティール", "Teal"],
  "settings.title": ["设置分离任务", "分離の設定", "Set up separation"],
  "settings.input": ["输入音频", "入力音声", "Input audio"],
  "settings.inputPlaceholder": ["选择需要处理的音频", "処理する音声ファイルを選択", "Choose audio to process"],
  "settings.chooseFile": ["选择文件", "ファイルを選択", "Choose file"],
  "settings.chooseDirectory": ["选择目录", "フォルダーを選択", "Choose folder"],
  "settings.model": ["分离模型", "分離モデル", "Separation model"],
  "settings.models": ["模型目录", "モデルフォルダー", "Model folder"],
  "settings.modelsPlaceholder": ["包含原始权重文件的文件夹", "元のモデルファイルが入ったフォルダー", "Folder containing original model files"],
  "settings.output": ["输出目录", "出力フォルダー", "Output folder"],
  "settings.outputPlaceholder": ["选择分离音轨的保存位置", "分離したトラックの保存先", "Choose where to save separated tracks"],
  "settings.weights": ["所需文件：{filename}", "必要なファイル：{filename}", "Required file: {filename}"],
  "settings.format": ["双声道 · 44.1 kHz · 32 位浮点 WAV", "ステレオ · 44.1 kHz · 32 bit 浮動小数点 WAV", "Stereo · 44.1 kHz · 32-bit float WAV"],
  "model.1296.label": ["人声与伴奏 · BS-RoFormer 1296", "ボーカルと伴奏 · BS-RoFormer 1296", "Vocals & instrumental · BS-RoFormer 1296"],
  "model.deecho.label": ["去回声与混响 · DeEcho", "エコー・残響除去 · DeEcho", "Echo & reverb removal · DeEcho"],
  "model.1296.note": ["输出人声与伴奏两轨，保留原始增益。", "元の音量を保ちながら、ボーカルと伴奏を出力します。", "Exports vocals and instrumental tracks while preserving the original gain."],
  "model.5hp.note": ["输出卡拉 OK 侧与移除侧两轨，可分别试听确认。", "カラオケ側と除去成分を出力します。それぞれ試聴して確認できます。", "Exports the karaoke side and removed component for separate listening."],
  "model.6hp.note": ["另一套 Karaoke 权重，输出卡拉 OK 侧与移除侧两轨。", "別の Karaoke モデルで、カラオケ側と除去成分を出力します。", "An alternative Karaoke model that exports the karaoke side and removed component."],
  "model.deecho.note": ["输出去混响音频，以及残余回声／混响两轨。", "残響を抑えた音声と、取り除いたエコー・残響を出力します。", "Exports dereverberated audio and the residual echo/reverb."],
  "runtime.title": ["推理运行时", "推論ランタイム", "Inference runtime"],
  "runtime.reset": ["恢复默认", "初期設定に戻す", "Reset defaults"],
  "runtime.backend": ["计算后端", "実行バックエンド", "Compute backend"],
  "runtime.threads": ["计算线程数", "計算スレッド数", "Compute threads"],
  "runtime.advanced": ["高级推理参数", "推論の詳細設定", "Advanced inference settings"],
  "runtime.windowFrames": ["窗口帧数", "ウィンドウのフレーム数", "Window frames"],
  "runtime.batch": ["推理批量", "推論バッチサイズ", "Inference batch"],
  "runtime.parallel": ["并行窗口数", "並列ウィンドウ数", "Parallel windows"],
  "runtime.timeBatch": ["时间注意力批量", "時間方向 Attention のバッチ", "Time-attention batch"],
  "runtime.frequencyBatch": ["频率注意力批量", "周波数方向 Attention のバッチ", "Frequency-attention batch"],
  "runtime.layout": ["线性层布局", "線形層のレイアウト", "Linear layer layout"],
  "runtime.flattenedDefault": ["展平 · 默认", "フラット化 · 初期設定", "Flattened · Default"],
  "runtime.flattened": ["展平布局", "フラット化レイアウト", "Flattened layout"],
  "runtime.batched": ["分批布局", "バッチレイアウト", "Batched layout"],
  "runtime.default": [" · 默认", " · 初期設定", " · Default"],
  "runtime.roformerHint": ["批量控制每次计算的注意力分组数；并行窗口数为 1–8。增大数值会提高内存需求。", "バッチサイズは一度に処理する Attention グループ数です。並列ウィンドウ数は 1～8。値を大きくするとメモリ使用量が増えます。", "Batch sizes control attention groups per call; parallel windows range from 1–8. Larger values require more memory."],
  "runtime.deechoHint": ["DeEcho 固定使用批量 1、并行窗口 1。", "DeEcho はバッチサイズ 1、並列ウィンドウ数 1 で動作します。", "DeEcho uses batch size 1 and one window at a time."],
  "runtime.batchHint": ["批量大于 1 时按批处理，并行窗口固定为 1。", "バッチサイズが 1 より大きい場合、並列ウィンドウ数は 1 に固定されます。", "With batch sizes above 1, batches run with window concurrency fixed at 1."],
  "runtime.parallelHint": ["同时处理多个窗口；增加批量或并行窗口会提高内存需求。", "複数のウィンドウを同時に処理します。バッチサイズや並列数を増やすとメモリ使用量が増えます。", "Process several windows together. Larger batches or more parallel windows require more memory."],
  "runtime.windowHint": ["{min}–2048 帧，须为 16 的倍数。窗口大小可能影响分离结果。", "{min}～2048 フレーム、16 の倍数で指定します。サイズによって分離結果が変わる場合があります。", "{min}–2048 frames, in multiples of 16. Window size may affect separation results."],
  "runtime.openvinoHint": ["OpenVINO CPU 使用自身的计算调度，可调整计算线程数。", "OpenVINO CPU は専用の実行スケジューラーを使用します。計算スレッド数を調整できます。", "OpenVINO CPU manages its own scheduling. You can adjust its compute thread count."],
  "runtime.burnHint": ["调整线程、批量和并行窗口以适配本机 CPU 与可用内存。", "CPU と使用可能なメモリに合わせて、スレッド数、バッチサイズ、並列数を調整できます。", "Tune threads, batches and parallel windows for your CPU and available memory."],
  "runtime.vrHint": ["当前 VR 模型使用 Burn CPU 运行。", "この VR モデルは Burn CPU で動作します。", "This VR model runs on Burn CPU."],
  "runtime.invalid": ["请检查运行参数，数值须在允许范围内。", "推論設定を確認してください。値は指定範囲内で入力します。", "Check the runtime settings. Values must be within the allowed ranges."],
  "runtime.summary.threads": ["{count} 线程", "{count} スレッド", "{count} threads"],
  "runtime.summary.frames": ["{count} 帧／窗", "{count} フレーム／ウィンドウ", "{count} frames/window"],
  "runtime.summary.batch": ["批量 {count}", "バッチ {count}", "Batch {count}"],
  "runtime.summary.parallel": ["{count} 个并行窗口", "並列ウィンドウ {count}", "{count} parallel windows"],
  "runtime.summary.time": ["时间批量 {count}", "時間バッチ {count}", "Time batch {count}"],
  "runtime.summary.frequency": ["频率批量 {count}", "周波数バッチ {count}", "Frequency batch {count}"],
  "task.title": ["处理进度", "処理状況", "Processing progress"],
  "task.start": ["开始分离 →", "分離を開始 →", "Start separation →"],
  "task.starting": ["正在分离…", "分離中…", "Separating…"],
  "task.cancel": ["取消任务", "処理をキャンセル", "Cancel task"],
  "task.cancelling": ["正在取消…", "キャンセル中…", "Cancelling…"],
  "task.status.waiting": ["等待开始", "開始待ち", "Ready to start"],
  "task.status.running": ["正在处理", "処理中", "Processing"],
  "task.status.completed": ["已完成", "完了", "Completed"],
  "task.status.cancelled": ["已取消", "キャンセル済み", "Cancelled"],
  "task.status.failed": ["处理失败", "処理に失敗", "Failed"],
  "task.ready": ["准备好音频后即可开始", "音声ファイルを選んで開始してください", "Choose your audio to get started"],
  "task.progressLabel": ["当前阶段进度", "現在のステージの進行状況", "Current stage progress"],
  "task.elapsed": ["已用时", "経過時間", "Elapsed"],
  "task.outputs": ["输出音轨", "出力トラック", "Output tracks"],
  "task.outputCount": ["{count} 个文件", "{count} ファイル", "{count} files"],
  "task.empty": ["完成后，两条音轨的保存位置会显示在这里。", "完了すると、2 つのトラックの保存先がここに表示されます。", "The locations of both saved tracks will appear here when processing finishes."],
  "task.log": ["任务记录", "処理ログ", "Task log"],
  "task.noLog": ["尚无任务。", "まだ処理はありません。", "No tasks yet."],
  "task.note": ["处理速度取决于音频长度与模型。取消时会等待当前计算片段结束，已有同名输出会保留。", "処理時間は音声の長さとモデルによって変わります。キャンセルは現在の計算区間の終了を待ちます。同名の既存ファイルは上書きしません。", "Processing time depends on audio length and model. Cancellation waits for the current compute segment; existing output files are preserved."],
  "task.runtime": ["运行配置：{configuration}", "実行設定：{configuration}", "Runtime settings: {configuration}"],
  "task.phase.prepare": ["准备任务", "処理を準備中", "Preparing task"],
  "task.phase.decode": ["读取音频", "音声を読み込み中", "Reading audio"],
  "task.phase.loadModel": ["核验并载入模型", "モデルを検証・読み込み中", "Verifying and loading model"],
  "task.phase.analysis": ["分析音频", "音声を解析中", "Analyzing audio"],
  "task.phase.inference": ["分离音频 · {completed}/{total} 窗口", "音声を分離中 · {completed}/{total} ウィンドウ", "Separating audio · {completed}/{total} windows"],
  "task.phase.reconstruction": ["重建音轨", "トラックを再構成中", "Reconstructing tracks"],
  "task.phase.encode": ["保存 WAV", "WAV を保存中", "Saving WAV files"],
  "task.phase.complete": ["完成 · {sampleRate} Hz · {samples} 个采样 / 声道", "完了 · {sampleRate} Hz · {samples} サンプル／チャンネル", "Complete · {sampleRate} Hz · {samples} samples/channel"],
  "task.phase.cancelled": ["已取消，未完成的输出已清理", "キャンセルしました。未完了の出力は削除されました", "Cancelled; incomplete outputs were removed"],
  "task.phase.error": ["处理失败：{error}", "処理に失敗しました：{error}", "Processing failed: {error}"],
  "track.vocals": ["人声", "ボーカル", "Vocals"],
  "track.instrumental": ["伴奏", "伴奏", "Instrumental"],
  "track.dry": ["去混响音频", "残響除去済み音声", "Dereverberated audio"],
  "track.reverb": ["残余回声／混响", "除去したエコー・残響", "Residual echo/reverb"],
  "track.karaoke": ["卡拉 OK 侧", "カラオケ側", "Karaoke side"],
  "track.removed": ["移除侧", "除去成分", "Removed component"],
  "track.output": ["输出音轨", "出力トラック", "Output track"],
  "library.title": ["模型管理", "モデル管理", "Model library"],
  "library.initial": ["选择目录后检测", "フォルダーを選んで確認", "Choose a folder to check"],
  "library.note": ["模型保存在所选目录，支持目录软链接。也可以把 models 文件夹与程序放在一起携带。", "モデルは選択したフォルダーに保存されます。シンボリックリンクにも対応しています。models フォルダーをアプリと一緒に持ち運ぶこともできます。", "Models are stored in the selected folder, with folder symlinks supported. You can also keep a models folder beside the app."],
  "library.refresh": ["重新检测", "再確認", "Check again"],
  "library.reset": ["使用默认目录", "既定のフォルダーを使う", "Use default folder"],
  "library.checking": ["正在检测模型…", "モデルを確認中…", "Checking models…"],
  "library.verifying": ["正在校验模型文件…", "モデルファイルを検証中…", "Verifying model files…"],
  "library.available": ["当前模型已就绪", "選択したモデルは使用できます", "The selected model is ready"],
  "library.unavailable": ["当前模型尚未就绪，可在下方模型管理中检测或下载。", "選択したモデルはまだ使用できません。下のモデル管理で確認またはダウンロードしてください。", "The selected model is not ready. Check or download it in the model library below."],
  "library.count": ["{count} / {total} 个模型已就绪", "{total} モデル中 {count} 個が使用可能", "{count} / {total} models ready"],
  "library.ready": ["已校验，可直接使用", "検証済み・使用可能", "Verified and ready"],
  "library.missing": ["尚未下载", "未ダウンロード", "Not downloaded"],
  "library.invalid": ["模型文件未通过校验", "モデルファイルの検証に失敗", "Model file failed verification"],
  "library.error": ["无法读取模型文件", "モデルファイルを読み込めません", "Cannot read model file"],
  "library.actionReady": ["已就绪", "使用可能", "Ready"],
  "library.redownload": ["重新下载", "再ダウンロード", "Download again"],
  "library.download": ["下载", "ダウンロード", "Download"],
  "download.source": ["下载来源", "ダウンロード元", "Download source"],
  "download.proxy": ["代理地址", "プロキシアドレス", "Proxy address"],
  "download.optional": ["可选", "任意", "Optional"],
  "download.proxyPlaceholder": ["http://127.0.0.1:7890 或 socks5h://…", "http://127.0.0.1:7890 または socks5h://…", "http://127.0.0.1:7890 or socks5h://…"],
  "download.proxyHint": ["留空使用系统／环境代理设置。", "空欄の場合はシステム・環境変数のプロキシ設定を使用します。", "Leave blank to use system/environment proxy settings."],
  "download.initial": ["按需下载，校验通过后即可使用。", "必要なモデルをダウンロードしてください。検証後に使用できます。", "Download models as needed; they are ready to use after verification."],
  "download.prepare": ["准备下载 {model}", "{model} のダウンロードを準備中", "Preparing to download {model}"],
  "download.cancel": ["取消下载", "ダウンロードをキャンセル", "Cancel download"],
  "download.progressLabel": ["模型下载进度", "モデルのダウンロード状況", "Model download progress"],
  "download.phase.checking": ["检查已有模型", "既存のモデルを確認中", "Checking existing model"],
  "download.phase.connecting": ["正在连接下载源", "ダウンロード元に接続中", "Connecting to download source"],
  "download.phase.receiving": ["正在下载模型", "モデルをダウンロード中", "Downloading model"],
  "download.phase.verifying": ["校验下载文件", "ダウンロードしたファイルを検証中", "Verifying downloaded file"],
  "download.phase.ready": ["模型已就绪", "モデルは使用できます", "Model ready"],
  "download.phase.completed": ["下载完成，校验通过", "ダウンロード・検証が完了しました", "Download complete and verified"],
  "download.phase.existing": ["模型已存在，校验通过", "既存のモデルの検証が完了しました", "Existing model verified"],
  "download.phase.cancelled": ["下载已取消，临时文件已清理", "ダウンロードをキャンセルし、一時ファイルを削除しました", "Download cancelled; temporary files were removed"],
  "download.phase.error": ["下载失败：{error}", "ダウンロードに失敗しました：{error}", "Download failed: {error}"],
  "footer.local": ["原始权重 · 离线处理", "元のモデルファイル · オフライン処理", "Original checkpoints · Offline processing"],
  "footer.scope": ["单模型工作台 · 多模型处理链正在接入", "単一モデルのワークスペース · 複数モデルの連続処理は開発中", "Single-model workspace · Multi-model chains in development"],
};

const storageKey = "uvr.language.v1";
const languages: Language[] = ["zh-CN", "ja", "en"];
function initialLanguage(): Language {
  try {
    const saved = localStorage.getItem(storageKey);
    if (languages.includes(saved as Language)) return saved as Language;
  } catch { /* Language selection also works when storage is unavailable. */ }
  for (const candidate of navigator.languages ?? [navigator.language]) {
    if (candidate.toLowerCase().startsWith("zh")) return "zh-CN";
    if (candidate.toLowerCase().startsWith("ja")) return "ja";
    if (candidate.toLowerCase().startsWith("en")) return "en";
  }
  return "zh-CN";
}
let language = initialLanguage();
export function getLanguage(): Language { return language; }
export function t(key: string, args: MessageArgs = {}): string {
  const template = messages[key]?.[languages.indexOf(language)] ?? key;
  return template.replace(/\{(\w+)\}/g, (placeholder, name: string) => String(args[name] ?? placeholder));
}

export interface TranslatedPhase { phase: string; phaseKey?: string; phaseArgs?: MessageArgs }
export function phaseText(event: TranslatedPhase): string {
  return event.phaseKey && messages[event.phaseKey] ? t(event.phaseKey, event.phaseArgs) : event.phase;
}

function translateDocument(): void {
  document.documentElement.lang = language;
  document.title = t("app.title");
  document.querySelectorAll<HTMLElement>("[data-i18n]").forEach(node => {
    node.textContent = t(node.dataset.i18n!);
  });
  for (const attribute of ["placeholder", "title", "aria-label"] as const) {
    document.querySelectorAll<HTMLElement>(`[data-i18n-${attribute}]`).forEach(node => {
      node.setAttribute(attribute, t(node.getAttribute(`data-i18n-${attribute}`)!));
    });
  }
  const select = document.getElementById("language") as HTMLSelectElement | null;
  if (select) select.value = language;
}

export function setLanguage(value: Language): void {
  if (!languages.includes(value)) return;
  language = value;
  try { localStorage.setItem(storageKey, value); } catch { /* Storage is optional. */ }
  translateDocument();
  window.dispatchEvent(new Event("uvr-languagechange"));
}
export function onLanguageChange(callback: () => void): void {
  window.addEventListener("uvr-languagechange", callback);
}
export function initializeLanguage(): void {
  translateDocument();
  document.getElementById("language")?.addEventListener("change", event => {
    setLanguage((event.target as HTMLSelectElement).value as Language);
  });
}
