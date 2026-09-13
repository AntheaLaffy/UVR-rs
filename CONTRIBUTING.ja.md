# UVR Rust への貢献

[English](CONTRIBUTING.md) · [简体中文](CONTRIBUTING.zh-CN.md) · 日本語

ローカルでの音声分離を使いやすく、結果を検証しやすくするための開発に参加できます。まず[ユーザーガイド](README.ja.md)と[ランタイムガイド](docs/runtime.ja.md)を参照してください。推論の変更では音声の扱いと品質の許容誤差を維持します。高速化は、その条件を守って実際のタスクを改善したときに意味があります。

## 変更する場所

| 場所 | 責務 |
| --- | --- |
| `core/` | 音声デコード／DSP、元の重みの読み込み、推論、ファイル処理 |
| `cli/` | 引数解析、環境変数の互換性、進捗表示、終了コード |
| `gui/src/` | 操作、入力検証、設定の保存、進捗表示 |
| `gui/src-tauri/` | ネイティブコマンド、モデル管理、タスクの制御 |
| `tools/reference/` | 独立した参照データの生成と検証。製品実行時には使用しない |
| `benchmarks/` | 再現可能な正しさと性能の実験 |
| `upstream/`、`agent/deepseek-harness/` | 製品 workspace 外の固定バージョンの参照 |

Cargo workspace は既定で core と CLI を対象とし、CLI 開発にデスクトップ依存関係を要求しません。pnpm workspace は GUI を管理します。通常のビルドで参照サブモジュールは不要です。上流コードが必要な作業でのみ、[参照資料](docs/references.md)に従って初期化してください。

推論と音声の処理は core に置きます。CLI と GUI は `RuntimeOptions` と `separate_file_with_options` を共用し、検証、実際の設定、タスクごとのスレッドプールを一元化します。選んだモデルとバックエンドが実装している設定だけを表示してください。プロセス全体の環境変数を変更せず、CLI は互換用の環境変数を明示的な設定に変換し、GUI は選択された設定を渡します。

core の `burn-cpu` は推論と PCM タスク、`audio-io` はコーデックとファイル処理を提供し、CLI は両方を有効にします。`openvino` は任意で、対応するネイティブ CPU ランタイムが必要です。Python／PyTorch は独立検証専用です。通常のビルド、起動、分離処理から呼び出してはいけません。通常の Cargo テストは固定済みデータを使うため、Python 環境は不要です。

## 検証する

Rust 2024 対応ツールチェーンを用意します。デスクトップ開発には Node.js 24、pnpm 12.1.0、[Tauri のシステム依存関係](https://v2.tauri.app/start/prerequisites/)も必要です。依存関係は `pnpm install --frozen-lockfile` で導入します。

作業中は関連する検証を行い、引き渡し前に適用対象の workspace 検証を完了してください。

```sh
cargo fmt --all -- --check
cargo test --locked -p uvr-core -p uvr-cli
cargo test --locked -p uvr-core -p uvr-cli --features uvr-core/burn-cpu
cargo clippy --locked -p uvr-core --all-targets --features burn-cpu -- -D warnings
pnpm build
cargo check --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo run --locked -p uvr-cli -- --help
cargo run --locked -p uvr-cli -- --version
git diff --check
```

バックエンドやランタイムの変更では、任意の構成も確認します。

```sh
cargo test --locked -p uvr-core -p uvr-cli --features uvr-core/openvino,uvr-cli/openvino
cargo check --workspace --all-targets --locked --all-features
```

core の開発は `cargo check --locked` から始められます。通常のテストはハッシュ、DSP／コーデック、キャンセル、出力保護、設定検証、CLI の失敗ケースを扱います。大きなモデル重みがない環境でこれらが通っても、モデル全体の受け入れ検証にはなりません。[バックエンドプローブ](benchmarks/backend-probe/README.md)と[独立検証ツール](tools/reference/README.md)でネットワークや波形を比較し、実施したことと未検証の範囲を明記してください。

GUI はブラウザーでのプレビューに加えてネイティブアプリでも確認します。影響がある場合、モデル／バックエンドの切り替え、入力検証、設定保存、キャンセル、出力処理を確認してください。Web ビルドの成功だけではネイティブコマンドや推論の動作を確認できません。

Windows 向けの作業は Linux での開発と、協力者の Windows マシンでのビルド・実行確認を組み合わせられます。Windows の受け入れ検証は未完了です。性能差をクロスコンパイルに帰する前に、release 最適化、CPU 命令ターゲット、ツールチェーン、バックエンド設定、スレッドのスケジューリングを比較してください。`tools/build-native.mjs` は Linux 専用です。ビルドマシンの `target-cpu=native` を汎用の Windows 配布物に持ち込まないでください。

## 問題報告と変更提案

報告には利用者から見える問題と最短の再現方法を記載します。アプリの revision とビルド方法、OS／CPU、モデル、実際の推論設定、入力形式と長さ、期待した動作、関連するエラーやログを含めてください。性能の報告には入力と重みのハッシュ、計時範囲、繰り返し測定も必要です。私的な音声や大きな重みは添付せず、同じ問題を再現できる場合は共有可能な合成音を使います。

変更は一つの問題を単位にしてレビュー可能にします。PR には理由、変更後の動作、重要な取捨選択、検証の限界を記載します。共通設定を変更したら CLI ヘルプ、GUI、ランタイム文書を合わせて更新してください。英語、中国語、日本語の画面とユーザー文書を同期し、コマンド名、パス、引数の値は翻訳しません。表示テキストやスタイルを変えたら、言語切り替えとライト／ダーク表示を確認します。

依存関係を変更したら `Cargo.lock` と `pnpm-lock.yaml` も更新します。元の重み、利用者の音声、出力トラック、大きな実験ファイルは無視対象のディレクトリに置きます。共有する一覧と結果の要約には入力ハッシュを含めてください。参照サブモジュールの更新には新しい commit と理由が必要です。

## 証拠を残す

研究文書は現在主に中国語です。推論の前提を変える前に[タスク](docs/tasks.md)、[基準](docs/baseline.md)、[事前の知識](docs/prior-knowledge.md)を読みます。[検証済み知見](docs/posterior-knowledge.md)には日付、コード revision、証拠の場所、結果、適用範囲を記録します。

- 利用者の要件と実験で確認した結果を区別する。
- 未検証の判断は検証方法とともに事前の知識に残す。
- [性能検証手順](docs/performance.md)に従い、モデル、入力、品質基準、計時範囲を比較可能にする。局所的な演算やビルドの速さをタスク全体の推論性能と混同しない。
- 失敗や更新前の結論を残し、なぜ無効になったか説明する。回帰を隠すために許容誤差を広げない。

詳しい実験履歴は[ベンチマーク記録](benchmarks/README.md)に置きます。README は初めての利用者が選択して使い始めるための文書です。

## 共通のアイコン

デスクトップアイコン、ロゴ、favicon は元の `gui/src-tauri/icons/source.svg` を使います。濃い青緑の背景に薄緑の波形を描いた画像です。デスクトップの生成物は `gui/src-tauri/icons/uvr/`、Web 用は `gui/public/branding/uvr.png` と `uvr-32.png` に置き、同じデザインを維持します。

元画像の変更後は Tauri で書き出し、本プロジェクトで使うファイルだけをコピーしてください。

```sh
pnpm --filter @uvr/gui tauri icon src-tauri/icons/source.svg --output /tmp/uvr-icons
cp /tmp/uvr-icons/{32x32.png,64x64.png,128x128.png,128x128@2x.png,icon.png,icon.ico,icon.icns} gui/src-tauri/icons/uvr/
cp /tmp/uvr-icons/128x128.png gui/public/branding/uvr.png
cp /tmp/uvr-icons/32x32.png gui/public/branding/uvr-32.png
```
