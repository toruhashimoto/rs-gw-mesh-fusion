# MeshFusion（日本語版）

**RealityScan の High Detail メッシュと、3DGS 由来の補完メッシュを1つのモデルに融合し、RealityScan で「単一モデルとして」テクスチャ生成するためのツール。**

[English README](README.md)

RealityScan はモデルのインポートはできますが、複数モデルを1つに合成することは
できず、テクスチャもモデルごとに生成されます。MeshFusion は、RealityScan の
**High Detail メッシュを主表面として無劣化のまま保持**し、同じ写真セットから
[Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping) などで生成した
補完メッシュ（薄物・低テクスチャ領域に強い）から **RS に無い領域だけをパッチとして
切り出して追加**、**単一の融合 PLY** を出力します。これを RealityScan に再インポート
すれば、1つのモデルとしてテクスチャ生成できます。

- **融合コアは CPU のみ**（距離計算 = Open3D embree BVH、ICP、パッチ選定すべて CPU。GPU 不要）
- **非破壊** — 入力メッシュは変更しない。RS メッシュは常に不動（補完メッシュ側を位置合わせ）
- **半自動運用前提** — 毎回レポートとプレビュー（グレー=RS / オレンジ=補完）を出力し、
  確認 → パラメータ調整 → 再実行のループで追い込める
- RealityScan 3,220万面 × Gaussian Wrapping 615万面の実データで検証済み

## インストール

```bash
git clone https://github.com/toruhashimoto/rs-gw-mesh-fusion
cd rs-gw-mesh-fusion
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Windows / Python 3.10 以上を想定。

## 使い方

**デスクトップアプリ（ローカル Gradio UI）**: `launch_app.bat` をダブルクリック、または:

```bash
python app.py
```

**CLI**:

```bash
run_fuse.bat --rs rs_high_detail.ply --gw complement.ply --out out\fusion --icp
```

出力（`--out` 内）: `fused.ply`（単一モデル・頂点カラー保持）/ `fusion_report.txt` /
`fusion_meta.json` / `patch_faces.npy` /（`--obj` 指定時）`fused.obj`

その後 RealityScan で **Import Model** → インポートしたモデルを選択して **Texture**。
非マニホールドエッジの警告が出た場合は RS のクリーニングで問題ありません
（`--clean` で縮退面・重複面の事前除去も可能）。

## `--icp` が事実上必須な理由

**RealityScan のメッシュエクスポートと COLMAP エクスポートは座標系が一致しません**
（実測: 原点周りの微小回転、テストシーンで中央値 5.6 ユニットのズレ。スケールは一致）。
3DGS の再構成は COLMAP 座標系に乗るため、融合前に補完メッシュを RS 座標系へ
位置合わせする必要があります。MeshFusion は多段 ICP（対応距離を実測ズレ→RS エッジ長へ
段階縮小、前半は剛体・最終段のみスケール推定）でサブエッジ精度まで収束させます。
`--icp` なしで座標系不一致を検出した場合は、壊れた出力を作らず明確なエラーで停止します。

## 処理の中身

1. **位置合わせ検証** — RS 頂点→補完メッシュ表面の距離統計。不一致なら停止（`--icp` で自動補正）
2. **パッチ選定** — RS 表面から `tau`（既定: RS 中央値エッジ長×8）超の補完メッシュ面を、
   ROI（既定: RS バウンディングボックス+10%。`--roi_json` で凸包指定可 =
   Gaussian Wrapping の Blender アドオン形式互換）内に限定 → 微小破片成分を除去 →
   境界を数リング膨張して RS と重ねる（投影テクスチャでは重なりは無害）
3. **融合** — RS + パッチを単一 PLY に連結。トポロジは意図的に非連結のまま
   （RealityScan の投影テクスチャは watertight 連結メッシュを要求しないため）

## チューニング

| 症状 | 対処 |
|---|---|
| 補完が多すぎる（部屋の壁・床まで付く） | `--roi_json` で対象物の箱を指定、または `--roi_expand 0` |
| 細かい破片が残る | `--min_patch_area_ratio` を上げる（例 `1e-3`） |
| RS の穴が埋まらない | `--tau_factor` を下げる（例 `4`） |
| 座標系エラーで停止 | `--icp` を付ける（常時推奨） |
| RS の非マニホールド警告 | `--clean` を付ける + RS 側クリーニングを実行 |

## パイプライン全体

撮影 → COLMAP エクスポート → Gaussian Wrapping 学習・メッシュ抽出
（Windows / RTX 50 系のビルド知見 `NVCC_APPEND_FLAGS=-DUSE_CUDA` 等を含む）→
融合 → RealityScan テクスチャ生成の全手順は
[docs/pipeline-guide.ja.md](docs/pipeline-guide.ja.md) を参照。

## ライセンス

Gaussian-Splatting License に準拠した非商用・研究評価限定ライセンス
（[LICENSE.md](LICENSE.md)）。Gaussian Wrapping / gaussian-splatting / nvdiffrast
（プレビュー用のオプション依存）は同梱しておらず、それぞれのライセンスに従ってください。
