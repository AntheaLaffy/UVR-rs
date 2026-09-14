<a href="https://github.com/IronHpc"><img src="https://avatars.githubusercontent.com/u/328778207?v=4&amp;s=128" alt="IronHPC" width="64" height="64"></a>

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

Linux x86_64 で最適化ビルドを作る場合は、まず `make` と[ランタイムガイド](docs/runtime.ja.md)に記載された依存関係を用意し、次を実行します。

```sh
pnpm install --frozen-lockfile
make
```

最適化された実行ファイルは `target/native/release/` に生成されます。OpenVINO を使う場合は隣接する `lib/` も一緒に移動してください。Burn-only 版には `make native-burn` を使います。汎用ビルド、Windows、CI 成果物、依存関係の詳細は[ランタイムガイド](docs/runtime.ja.md)にまとめています。開発ビルドは [GitHub Actions](https://github.com/AntheaLaffy/UVR-rs/actions/workflows/build.yml)からも取得できます。

`make` が利用できない場合は、[ランタイムガイド](docs/runtime.ja.md)の基礎コマンドを直接実行してください。

## Rust crate の再利用

信号処理には [`uvr-dsp`](https://crates.io/crates/uvr-dsp)、モデルの識別・ダウンロードには [`uvr-models`](https://crates.io/crates/uvr-models)、一方の推論系列だけが必要な場合は [`uvr-vr`](https://crates.io/crates/uvr-vr) または [`uvr-roformer`](https://crates.io/crates/uvr-roformer)、ファイル処理全体には [`uvr-runtime`](https://crates.io/crates/uvr-runtime) を選びます。これらの API は共通実装の [`uvr-core`](https://crates.io/crates/uvr-core) を利用します。

## 開発に参加する

[貢献ガイド](CONTRIBUTING.ja.md)に構成、検証コマンド、問題報告、証拠の扱いをまとめています。[ランタイムガイド](docs/runtime.ja.md)は設定と配布、[ベンチマーク記録](benchmarks/README.md)は測定結果と制約を扱います。

研究文書は現在主に中国語です：[タスクと受け入れ条件](docs/tasks.md)、[基準](docs/baseline.md)、[性能検証手順](docs/performance.md)、[検証済み知見](docs/posterior-knowledge.md)、[上流の参照資料](docs/references.md)。本プロジェクトは参照先のモデルとアルゴリズムに基づいています。

## ライセンス

UVR Rust 独自のコードと文書は [MIT License](LICENSE) で提供します。モデルの重みとその他の第三者素材には、それぞれの条件が適用されます。詳細は[第三者素材](THIRD_PARTY_NOTICES.md)を参照してください。
