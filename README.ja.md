# UVR Rust

[English](README.md) · [简体中文](README.zh-CN.md) · 日本語

**VR の CPU 推論を Python 参照実装の約 3 倍の速度で。ネイティブ CLI は 6.94 MB。**

CPU での処理速度とアプリの小ささを重視する UVR ユーザー向けの選択肢です。元の UVR モデルをそのまま読み込み、デスクトップアプリとスクリプトから使える CLI を提供します。音声処理に Python や PyTorch ランタイムを同梱する必要はありません。

| 注目点 | UVR Rust |
| --- | --- |
| VR の CPU 推論 | RTF 約 2、Python 側は約 6：約 3 倍の速度 |
| 現在の実行ファイルサイズ | CLI 6.94 MB、デスクトップアプリ 18.91 MB。重みと追加ライブラリを除く |
| 任意の 1296 高速化ライブラリ | OpenVINO は別途 86.80 MB。CLI と合わせて 93.74 MB、GUI と合わせて 105.71 MB |
| 元の重み | `.pth`／`.ckpt` を直接読み込み。変換不要 |
| 日常の操作 | ローカル処理、GUI と CLI、進捗、キャンセル、既存出力の保護 |

速度は同じマシンで記録した 5-HP の構成比較で、音源とスレッド数は異なります。同一リソースでの比較ではありません。RTF 2 は、10 秒の音声に約 20 秒かかることを意味します。サイズは 9 月 13 日のシンボルを除去した Linux native release の値で、合計にもモデルとシステムライブラリは含みません。GUI は推論を内蔵し、CLI の追加は不要です。正確なバイト数、ハッシュ、過去のビルドと測定条件は[性能とサイズの記録](docs/performance-summary.ja.md)を参照してください。

## モデルを選ぶ

| 目的 | モデル／CLI キー | 元の重みファイル | 出力 |
| --- | --- | --- | --- |
| ボーカルと伴奏の分離 | BS-RoFormer 1296／`1296` | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | `vocals`、`instrumental` |
| Karaoke 分離を試す | 5-HP／`5hp` | `5_HP-Karaoke-UVR.pth` | `primary`、`residual` |
| 別の Karaoke モデルと比較 | 6-HP／`6hp` | `6_HP-Karaoke-UVR.pth` | `primary`、`residual` |
| エコーやリバーブを減らす | DeEcho／`deecho` | `UVR-DeEcho-DeReverb.pth` | `primary`、`residual` |

Karaoke と DeEcho の `primary` はモデルの主出力、`residual` は補完マスクから再構成した音声です。実際の素材で両方を試聴してください。ラベルは完全な主旋律／コーラス、ドライ音／残響の分離を保証するものではありません。モデルはアプリとは別にダウンロードします。配布元とファイル名は[モデル一覧](references/targets.json)に記録しています。

単一モデルの CLI とデスクトップ処理を実装済みです。自動の複数モデル処理、対応プラットフォームの拡大、楽曲全体の品質・性能検証は進行中です。画面とユーザー向け文書は日本語、英語、簡体字中国語に対応しています。対応する重みは上記の 4 種類で、UVR の全モデルを扱うものではありません。

## 使い方

### デスクトップ

`uvr-gui` を起動し、入力音声とモデル、モデル保存先と出力先を選びます。モデル管理では既存ファイルを検証でき、不足するモデルを表示された配布元からダウンロードできます。モデルの準備ができれば、音声をオフラインで処理できます。

推論設定でバックエンドとスレッド数を、高度な設定で適用可能なウィンドウ、バッチ、レイアウトを変更できます。設定はローカルに保存され、選択中のモデルの推奨値に戻すこともできます。タスクログには実際に使った設定が残るため、CLI でも再現できます。両方の入口が同じ Rust の推論実装と検証規則を使用します。

言語、システムに合わせる／ライト／ダーク表示、アクセントカラーを選べます。言語と外観の設定もローカルに保存されます。

### コマンドライン

下記の通常ビルドを終えた後、リポジトリのルートから実行します。

```sh
./target/release/uvr --lang ja separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296

./target/release/uvr --lang ja separate-vr deecho \
  models/UVR-DeEcho-DeReverb.pth input.wav outputs/deecho

./target/release/uvr --lang ja --help
```

ネイティブビルドでは `target/native/release/uvr` を使います。両モデル系列で `--threads`、VR では `--window-frames`、`--inference-batch`、`--parallel-windows` を指定できます。1296 では `--backend` を選べ、Burn では注意機構のバッチ数、ウィンドウ並列数、線形層レイアウトも設定できます。値の範囲と例は[ランタイムガイド](docs/runtime.ja.md)を参照してください。

`--lang ja`、`--lang en`、`--lang zh-CN` はコマンドの前に指定します。`UVR_LANG` でも選べますが、明示的な引数が優先です。既定値は中国語です。機械可読の stdout キーは言語によらず同じで、下位層の技術的エラーは原文のまま表示します。

入力はモノラル／ステレオの WAV、FLAC、MP3。出力は 44.1 kHz、ステレオ、32 ビット浮動小数点 WAV の 2 トラックで、`<入力名>_<モデル>_<トラック>.wav` として保存します。正規化やクリッピングはしません。モノラルはステレオに複製し、異なるサンプリング周波数は変換します。既存の出力は上書きしません。Ctrl-C またはアプリのキャンセルボタンで停止できますが、応答時間は実行中の計算に依存します。CLI の終了コードは `0` 成功、`1` 処理失敗、`2` 引数エラー、`130` キャンセルです。

`inspect-audio <ファイル>` はデコード後の音声情報、`inspect-weights <ファイル>` はサイズ、SHA-256、UVR のメタデータ MD5 を表示します。ハッシュ計算だけでは、対応モデルであることを確認したことにはなりません。

## ビルドとインストール

開発ビルドは [GitHub Actions](https://github.com/AntheaLaffy/UVR-rs/actions/workflows/build.yml)から取得できます。push、PR、手動実行で起動し、成功した実行の成果物をダウンロードします。Linux の CLI／GUI と追加の OpenVINO CPU ライブラリは別々のパッケージです。Windows はネイティブの Burn ビルドで、実機での音声の受け入れ検証は未完了です。これらはワークフローの成果物であり、GitHub Release ではありません。

最初の検証対象は Linux x86_64 です。Rust 2024 edition に対応したツールチェーンが必要です。デスクトップには Node.js 24、pnpm 12.1.0、[Tauri のシステム依存関係](https://v2.tauri.app/start/prerequisites/)も必要です。通常のビルドと推論に参照用サブモジュールや Python は不要です。

Windows は未検証です。Linux での開発と、協力者による Windows 上のビルド・テストを並行して進められます。クロスコンパイル自体が推論を遅くするわけではなく、release 最適化、実行先 CPU の命令セット、ツールチェーン、ランタイム設定が影響します。別の PC 向けには互換性のある CPU ターゲットを選んでください。以下のネイティブビルド用スクリプトは Linux 専用です。

### この CPU 向けの推奨ビルド

[ランタイムガイド](docs/runtime.ja.md)に従って OpenVINO CPU のネイティブライブラリを用意し、リポジトリのルートで実行します。

```sh
pnpm install --frozen-lockfile
UVR_OPENVINO_LIB_DIR=/path/to/openvino/lib pnpm build:native
./target/native/release/uvr-gui
```

`pnpm gui:build:native` も同じビルドを実行します。出力は `target/native/release/uvr`、`uvr-gui`、隣接する `lib/` です。移動時は一緒に配置してください。1296 は利用可能なら OpenVINO CPU、それ以外は Burn を既定で選びます。VR は Burn を使用します。既定のスレッド数は利用可能な論理 CPU 数と 8 の小さい方です。

`target-cpu=native` で現在の Linux x86_64 CPU 向けに作るため、任意の PC に配布できるビルドではありません。Burn だけなら `pnpm build:native --burn-only` を実行します。CLI と GUI は OpenVINO 版とは別の `target/native-burn/release/` に生成され、OpenVINO ライブラリは不要です。

### 通常ビルド

デスクトップ依存関係や、この CPU 固有の命令を必要としない CLI ビルド：

```sh
cargo build --release --locked -p uvr-cli
```

標準のターゲット設定によるデスクトップビルド：

```sh
pnpm install --frozen-lockfile
pnpm gui:build
./target/release/uvr-gui
```

これらは既定で Burn を使用します。`pnpm build` は Web フロントエンドのみを生成し、ローカル音声の処理にはデスクトップホストが必要です。インストーラーの生成はまだ有効にしていません。開発には `pnpm gui:dev` を使います。

## 開発に参加する

[貢献ガイド](CONTRIBUTING.ja.md)に構成、検証コマンド、問題報告、証拠の扱いをまとめています。[ランタイムガイド](docs/runtime.ja.md)は設定と配布、[ベンチマーク記録](benchmarks/README.md)は測定結果と制約を扱います。

研究文書は現在主に中国語です：[タスクと受け入れ条件](docs/tasks.md)、[基準](docs/baseline.md)、[性能検証手順](docs/performance.md)、[検証済み知見](docs/posterior-knowledge.md)、[上流の参照資料](docs/references.md)。本プロジェクトは参照先のモデルとアルゴリズムに基づいています。
