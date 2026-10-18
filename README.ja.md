# ComfyUI 用 SyncReward

動画の**音と映像がどれだけ同期しているか**をスコア化し、複数の生成動画から最も同期の良いものを選ぶノード集です。公式の
**SyncReward**（[WeChatCV/SyncReward](https://github.com/WeChatCV/SyncReward)、論文
[arXiv:2610.16338](https://arxiv.org/abs/2610.16338)）を ComfyUI のプロセス内で動かし（サブプロセスなし、実行時ダウンロードなし）、
標準の `VIDEO` / `AUDIO` 型で扱います。

> **非公式**の統合です。SyncReward の著者（中国人民大学・WeChat / Tencent）、Meta（PE-AV）、Comfy Org とは無関係で、承認も受けて
> いません。詳しい英語版は [README.md](README.md)、検証の生データは [docs/VERIFICATION.md](docs/VERIFICATION.md) にあります。

## ノード

| ノード | 役割 |
| --- | --- |
| SyncReward Loader | Stage-2 チェックポイントと PE-AV エンコーダのフォルダ、デバイス、常駐の有無を選ぶ |
| SyncReward Score | `VIDEO`（音声つき。または `AUDIO` で差し替え）→ 0〜2 の報酬値（`FLOAT`）と JSON レポート |
| SyncReward Select Best | 2〜6 本の `VIDEO` から報酬が最も高いものを選び、その番号・報酬・全スコアも返す |
| SyncReward Shift Audio | 映像と長さを変えずに音声だけを時間方向にずらす（意図的にずれた試験用動画を作る） |
| SyncReward Unload | 常駐モデルの解放（GPU・RAM） |

報酬値は「聞こえる出来事と見える動作がどれだけ揃っているか」の**モデルによる予測**です（0＝同期していない、2＝よく同期）。
生成された音声つき動画への人手の評価で学習されています。見るのは動画中央の **4.75 秒の窓**（0.5 秒×18 セグメント）です。

## 何をどこまで確認したか（Windows 11 / RTX 5090 / ComfyUI v0.38.0、公式リポジトリのデモ動画 10 本）

- 同じ Python 環境（torch 2.9.1、transformers 5.1.0）で、このノードのデコード結果（フレーム、フレームレート、48 kHz モノラル音声）は
  公式 `load_av` と**ビット単位で一致**（10/10）、報酬値も公式モデルコードと**一致**（10/10）。
- ComfyUI の実キュー（torch 2.14.1 + transformers 5.20.1、クリーンインストール）では 10 本中 9 本が公式のコマンドライン結果と一致し、
  1 本は bfloat16 の 1 刻み（0.008）違いました（ライブラリのバージョン差が原因で、デコードではありません）。
- スコア、最良選択、ずらしてスコア、エラー表示、解放、CPU デバイスを実モデルで確認。CI（Ubuntu/Windows）は重みなしの
  スタブで CPU テストと、モデル不要ノードの実 ComfyUI キューを実行します。

- **論文のベンチマークを再現しました**（2026-10-18）：SyncReward-Bench の 1,000 本すべてをこのノードでスコアリングし、
  Spearman 0.781、Top-1（N=4）0.764、ペア正解率 0.835（モデルカードは 0.7823 / 0.7646 / 0.8359、95 % 区間内。同じマシンで公式の
  `evaluate.py` を動かすと 0.7810 / 0.7641 / 0.8351）。公式と同じ環境では、1,000 本すべてで報酬値の差が bfloat16 の 1 刻み
  （0.008）以内です。

**未確認:** GPU 付きの Linux / macOS、AMD・Apple GPU、実写（生成ではない）動画、約 5 秒でない動画、そして Select Best が想定する
「同じプロンプト・同じ生成モデルの複数候補」からの選択。

> **スコアや Select Best に頼る前に [LIMITATIONS.md](LIMITATIONS.md)（英語）を読んでください。** 人手評価が 1.5 以上のベンチマーク
> 動画 70 本での実測：音声を ±1 秒ずらしても報酬は平均 0.01 しか変わらず検出できません。無音・ノイズ・別動画の音声で明確に下がる
> のは 30〜70 % 程度の動画だけ、強い再エンコードや半分の解像度、16 kHz 音声は、音を消した場合の 55〜80 % ほど報酬を下げます。スコアは
> 平均としては人間の評価とよく一致しますが、**タイミングのずれの検出器でも、音の欠落のテストでもありません**。

## インストール

1. ComfyUI（V3 ノード API、v0.38.0 で確認）、Python 3.10 以上、空き VRAM 約 10 GB の NVIDIA GPU（CPU でも遅いが動作）。
2. このフォルダを `ComfyUI/custom_nodes/` に置き、要件を ComfyUI の Python に入れます（`transformers>=5.1.0`、`timm`、`PyYAML`）。

   ```
   python -m pip install -r custom_nodes/ComfyUI-SyncReward/requirements.txt
   ```
3. 2 つのモデル（合計約 15 GB、どちらもモデルカード上は Apache-2.0）を `models/syncreward/` にダウンロードします。

   ```
   python custom_nodes/ComfyUI-SyncReward/tools/download_models.py --models-dir ComfyUI/models/syncreward
   ```
4. ComfyUI を再起動し、`workflows/gui/` のワークフローを読み込みます（Load Video は `example.mp4` を指しているので自分の動画に変更）。

ノード自体は何もダウンロードしません。

## 実測値と注意点

- RTX 5090、bfloat16 autocast（公式 `score.py` と同じ）：読み込み約 10 秒、5 秒の動画のスコアリング約 1.6〜2 秒、モデルの VRAM
  ピーク 9.8 GiB（`encode_batch_size` 1 で 6.7 GiB）。CPU（fp32）では読み込み込みで 103 秒。
- **スコアは粗い指標です。** 10 本のデモ動画で音声をずらした平均報酬は、±1 秒でも 0.99〜1.03（ずらさない場合 1.023）と
  わずかしか下がらず、無音は 0.718、別の動画の音声は 0.824、2 秒ずらしで 0.951、3 秒ずらしで 0.879、逆再生は 1.031（検出できず）
  でした。小さなタイミングのずれや逆再生の検出には使わないでください。公式の「after」動画は 5 組すべてで「before」より高い値でした。
  ベンチマーク動画 70 本の大きな試験でも同じ結果です（[LIMITATIONS.md](LIMITATIONS.md)）。
- **Select Best は「提案」として扱ってください。** 人手評価が近い（差が評価 1 段階の 1/3 以下）動画の組では、報酬が人間と同じ順に
  並べるのは 69 % だけです。報酬の差が約 0.1 未満ならノイズと見てください。エンコード条件（解像度・ビットレート・フレームレート）
  が違う候補は、その違いでも選ばれます。
- transformers 5.20.1 は `facebook/pe-av-large` のネスト設定を 2 ラベルとして読み込み（5.1.0 は 1024）、そのままでは読み込みに
  失敗します。このノードは `config.json` の値に戻してから公式コードでモデルを構築します。
- 未加工の Load Video は、公式コードと同じ方法でファイルから直接デコードします（ComfyUI のデコーダは色タグのない動画で別の
  色変換を使い、報酬が最大 0.07 変わるため）。

## ライセンス

- このノード集のコード：**MIT**（[LICENSE](LICENSE)）。
- `syncreward_comfy/vendor/syncreward`：公式パッケージのコピー、**Apache-2.0**（[NOTICE](NOTICE)）。学習専用の関数 1 つ（`utils.py` の
  `setup_distributed`）だけを無効化しており（`syncreward_comfy/vendor/UPSTREAM.md`）、それ以外は無改変です。
- モデルの重み（SyncReward Stage 2、Meta の PE-AV）は**同梱していません**（モデルカードはいずれも Apache-2.0）。公式リポジトリの
  デモ動画も同梱しておらず、上記ライセンスの対象外です。

引用は公式リポジトリの BibTeX をご利用ください。
