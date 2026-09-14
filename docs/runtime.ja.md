# 推論ランタイムと設定

[English](runtime.md) | [简体中文](runtime.zh-CN.md) | 日本語

デスクトップアプリとCLIは同じRustの推論設定・検証処理を使います。UVRの元の重みを変換せずに、CPUスレッド、バッチ、並列ウィンドウを調整できます。バッチや並列数を増やすとメモリ使用量が増えるため、まずは推奨設定から始めてください。

## 推奨設定

| モデル | ランタイム | 既定値 |
| --- | --- | --- |
| 5-HP / 6-HP | 最適化されたVRカーネルを使うBurn CPU | 512フレーム、バッチ1、並列ウィンドウ4 |
| DeEcho | Burn CPU | 512フレーム、バッチ1、並列ウィンドウ1 |
| BS-RoFormer 1296 | 対応ビルドで使用可能ならOpenVINO CPU、それ以外はBurn | OpenVINO：FP32、遅延優先、1ストリーム。Burn：時間バッチ62、周波数バッチ301、flattenedレイアウト、並列ウィンドウ1 |

スレッド数は利用可能なCPU数と8の小さい方です。これらはIntel i5-13420Hでの測定に基づく開始点で、すべてのPCで最速になるとは限りません。ランタイムを切り替えても、元の重み、出力ゲイン、FP32ネットワークと既定の音声処理条件を保ちます。

OpenVINOの製品向け機能は現在1296のCPU推論のみです。実験用GPUは選択肢に含みません。アプリは使用可能なOpenVINO CPUを検出してから推奨し、利用できなければBurnを使います。使用できないランタイムを明示指定するとエラーになります。

### 未完了のチューニング

DeEchoのバッチ数とウィンドウ並列数のチューニングは未完了です。現在の実装では`inference_batch=1`、`window_parallelism=1`に固定しています。ウィンドウ長は変更可能で、既定値は512フレーム、範囲は144〜2048の16の倍数です。双方向LSTMの状態はウィンドウごとに独立させる必要があり、未来の文脈に依存するため、ウィンドウ間で状態を再利用しても同じ結果になるとは限りません。ただし、独立したウィンドウの並列実行が不可能という意味ではありません。1への固定は現在の実装の保守的な制限であり、状態の分離と出力検証を含むDeEchoの並列数1／2／4での音声処理全体の比較は未完了です。HPの測定からDeEchoの並列数を増やすと必ず遅くなるとは判断できません。

BS-RoFormer 1296も、BurnとOpenVINOの両方でチューニングが未完了です。Burnのレイアウトと注意機構のバッチ変更には計測・波形検証があり、OpenVINO CPUは短い音声で繰り返し比較し、GUI／CLIにも統合済みです。これらは途中段階の結果です。フルウィンドウの反復測定、最近のレイアウト変更の前後を交互に測る比較、曲全体と複数モデルの処理チェーンの性能検証が残っています。現在の既定値は調整の開始点であり、最速設定の探索が完了したことを示しません。残作業と採用条件は[性能測定手順（中国語）](performance.md#尚未完成的调优与验收)を参照してください。

## デスクトップアプリ

「推論ランタイム」で計算バックエンドとスレッド数を選び、詳細設定でモデル固有のパラメーターを調整します。設定概要とタスクのログには実際に適用する値を表示します。

- VRモデルごとのウィンドウ・バッチ設定と、1296のバックエンド・Burn設定を保存します。
- DeEchoはバッチ1・並列ウィンドウ1です。HPでバッチが1より大きいときはバッチ単位で処理し、個別ウィンドウの並列数は1になります。
- OpenVINOを選ぶと、Burn専用のバッチとレイアウト設定は非表示になります。
- 既定値に戻す操作で、選択中のモデルの推奨設定と既定スレッド数を復元できます。
- 実行中は推論設定をロックします。次のタスクは新しいスレッドプールを作るため、設定変更後のアプリ再起動は不要です。

表示言語は日本語・英語・簡体字中国語から選べます。実行中に言語を変えてもパス、設定、進捗は保持されます。外観はシステムに従う／ライト／ダークに加え、紫・青・青緑のテーマを選べます。言語と外観の設定は個別に保存されます。

## CLIオプション

推論オプションは必須の位置引数の後ろに指定し、順序は自由です。重複・不明なオプション、値の不足、適用できない設定は、ファイル処理前に終了コード2を返します。

言語はコマンドより前に指定します：`uvr --lang ja --help`。`en`、`zh-CN`も使えます。環境変数`UVR_LANG`より`--lang`を優先します。機械可読出力のキー、ファイル名、オプション値は翻訳しません。

| オプション | 対象 | 既定値・制約 |
| --- | --- | --- |
| `--threads N` | 全モデル | 利用可能なCPU数と8の小さい方。正の整数 |
| `--backend burn\|openvino-cpu` | `separate-1296` | 使用可能ならOpenVINO CPU、それ以外はBurn |
| `--window-frames N` | `separate-vr` | 512。16の倍数。HP：272–2048、DeEcho：144–2048 |
| `--inference-batch N` | `separate-vr` | 1。範囲1–4。DeEchoは実効値1 |
| `--parallel-windows N` | VRまたはBurn 1296 | 1–8。HP既定4、DeEchoと1296は1。HPバッチ>1では実効値1 |
| `--time-batch N` | Burn 1296 | 62。正の整数 |
| `--frequency-batch N` | Burn 1296 | 301。正の整数 |
| `--linear-layout flattened\|batched` | Burn 1296 | `flattened` |

RoFormerのバッチは独立したAttention系列をまとめる設定で、系列を短縮しません。バンド数やフレーム数を超える値は入力形状で制限されます。VRのウィンドウサイズ変更は分離結果に影響する場合があるため、既存の参照データと比較するときは既定値を使ってください。

```sh
target/native/release/uvr --lang ja separate-vr 5hp \
  models/5_HP-Karaoke-UVR.pth input.wav outputs/5hp \
  --threads 8 --window-frames 512 --inference-batch 1 --parallel-windows 4

target/native/release/uvr --lang ja separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296 \
  --backend openvino-cpu --threads 8

target/native/release/uvr --lang ja separate-1296 \
  models/model_bs_roformer_ep_368_sdr_12.9628.ckpt input.wav outputs/1296-burn \
  --backend burn --threads 8 --time-batch 62 --frequency-batch 301 \
  --parallel-windows 1 --linear-layout flattened
```

同名の出力ファイルは上書きしません。Ctrl-Cでキャンセルを要求すると終了コード130になります。VRは現在のウィンドウまたは並列グループの終了を待ちます。逐次実行のBurn 1296はウィンドウ内でもキャンセルを確認し、OpenVINOはネイティブリクエストをキャンセルします。

### 環境変数との優先順位

CLIは「明示したオプション > 対応する環境変数 > 既定値」の順です。オプションで上書きした環境変数は解析せず、OpenVINOではBurn専用の環境変数を無視します。

| 環境変数 | CLIオプション |
| --- | --- |
| `RAYON_NUM_THREADS` | `--threads` |
| `UVR_ROFORMER_TIME_BATCH` | `--time-batch` |
| `UVR_ROFORMER_FREQUENCY_BATCH` | `--frequency-batch` |
| `UVR_ROFORMER_WINDOW_PARALLELISM` | Burn 1296の`--parallel-windows` |
| `UVR_LINEAR_LAYOUT` | `--linear-layout` |

GUIは有効な`RAYON_NUM_THREADS`を初期スレッド数に使いますが、保存済み設定や画面で編集した値を優先します。他の推論設定は共有既定値と画面上の項目から取得し、見えない`UVR_*`設定で上書きしません。

## 開発機向けの最適化ビルド

```sh
pnpm install --frozen-lockfile
make
target/native/release/uvr-gui
```

このLinux x86_64向けビルドはrelease最適化と`-C target-cpu=native`を使い、CLIとGUIを`target/native/release/`に生成します。既定のOpenVINOライブラリ場所は`.local/openvino-2026.3.1/lib/`です。別の場所は`UVR_OPENVINO_LIB_DIR`で指定できます。`libopenvino_c.so`、CPUプラグインと依存ライブラリを用意してください。必要なCPUライブラリは実行ファイルの隣の`lib/`にコピーされます。

```sh
UVR_OPENVINO_LIB_DIR=/path/to/openvino/lib pnpm build:native
pnpm build:native --burn-only
target/native-burn/release/uvr-gui
```

`--burn-only`はOpenVINOなしで最適化されたBurn版を`target/native-burn/release/`に生成します。OpenVINO版の`target/native/release/`と分けることで、以前のビルドで配置した追加ライブラリが混ざるのを防ぎます。各ビルドの`release/build-info.json`には実行ファイルのサイズ、SHA-256、ビルド設定が記録されます。

今回のOpenVINO対応nativeビルドはCLIが6.94 MB、GUIが18.91 MBで、追加のOpenVINOライブラリは別途86.80 MBです。GUIとライブラリの合計は105.71 MBで、モデルとOSライブラリを含みません。正確なバイト数、ハッシュ、過去のビルドは[サイズの記録](performance-summary.ja.md)を参照してください。

`make` は同じビルドの短い入口で、内部では `pnpm build:native` を実行します。`make native-burn` は独立した Burn-only 版を生成します。実行時にPythonや変換スクリプトは不要です。OpenVINO版を移動するときは`lib/`も一緒に移動してください。モデルの重みは別途必要です。

`target-cpu=native`はビルドしたCPU向けです。他のCPUへ配布する汎用版には`pnpm gui:build`または`cargo build --release --locked -p uvr-cli`を使います。クロスコンパイル自体が推論を遅くするわけではありませんが、Windowsのツールチェーン・CPU命令・スケジューリングは実機で検証する必要があります。現在のネイティブビルドスクリプトはLinux専用で、Windows版はWindows環境でビルド・テストしてから配布してください。

測定の根拠は[性能とサイズ](performance-summary.ja.md)、貢献者向けの検証手順は[貢献ガイド](../CONTRIBUTING.ja.md)を参照してください。
