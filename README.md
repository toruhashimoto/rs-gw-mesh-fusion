# MeshFusion — RealityScan High Detail メッシュの GaussianWrapping 補完融合

RealityScan (RS) は複数モデルを1つに合成できず、テクスチャもモデル別になるため、
**RS の High Detail メッシュ（主）+ GaussianWrapping (GW) メッシュ（補完）を
1ファイルの単一モデルに融合**するツール。融合結果を RS にインポートすれば、
単一モデルとして RS でテクスチャ生成できる。

設計書: `docs/specs/2026-07-05-mesh-fusion-design.md` / 実装計画: `docs/plans/2026-07-05-mesh-fusion.md`

## 使い方

```bat
run_fuse.bat --rs <RSメッシュ.ply> --gw <GWメッシュ.ply> --out <出力Dir> --icp
```

- 実行環境は conda env `gaussian_wrapping`（run_fuse.bat が環境変数ごと設定）
- 入力は一切変更しない。RS メッシュは常に不動（座標補正は GW 側のみ）
- 出力: `fused.ply`（RS 座標系・頂点カラー付き）/ `fusion_report.txt` / `fusion_meta.json` /
  `patch_faces.npy` / `--obj` 指定時のみ `fused.obj`
- プレビュー（RS=グレー、補完パッチ=オレンジ）:

```bat
<env python> render_compare.py --fused <out>\fused.ply --meta <out>\fusion_meta.json --out <out>
```

## 重要な発見（Sample データで実証）

**RealityScan の「メッシュエクスポート」と「COLMAP エクスポート」は座標系が一致しない**
（Sample では原点周りの微小回転相当、中央値ズレ 5.6 ユニット。スケールは一致 1.00007）。
このため `--icp` は事実上必須。多段 ICP（対応距離を初期ズレ→エッジ長へ4段階縮小、
前半2段は剛体・後半のみスケール推定）で 中央値 0.010（RS エッジ長未満）まで収束する。

ICP のソース点は **RS のタイト AABB 内の GW 頂点のみ**を使う（AABB を拡張すると
GW の背景断片が混入してスケールが 0.87 に崩壊する — 実測済みの罠）。

## 処理フロー

1. **位置合わせ検証**: RS 頂点→GW 表面の距離統計。中央値 > `エッジ長×20` なら停止
   （`--icp` 指定時は多段 ICP 後に再判定）
2. **パッチ選定**: GW 面のうち RS 表面から `tau`（既定=RSエッジ長×8）超の面
   → ROI（既定=RS AABB+10%、`--roi_json` で Blender 凸包指定可）内に制限
   → 面積が RS 総面積×`1e-4` 未満の破片成分を除去 → 境界を3リング膨張して RS と重ねる
3. **結合**: RS + パッチを連結して単一 PLY 出力（トポロジは非連結のまま =
   RS の投影テクスチャには問題なし）

## チューニング指針

| 症状 | 対処 |
|---|---|
| 補完が多すぎる / 部屋の壁・床まで付く | `--roi_json`（Blender で対象物の箱を書いてエクスポート、GaussianWrapping の GW Bounds アドオン使用）で対象物に限定。または `--roi_expand 0` |
| 細かい破片が残る | `--min_patch_area_ratio` を上げる（例 `1e-3`） |
| 補完が足りない / RS の穴が埋まらない | `--tau_factor` を下げる（例 `4`）。GW 由来の面がより多く採用される |
| 座標系エラーで停止 | `--icp` を付ける（推奨: 常時付与） |

Sample_RS-ply での実測: パッチ 109万面（RS 面積比 39.7%）、うち78%は「GW だけが
再構成した部屋の大きな連続面」1成分。対象物だけ欲しい場合は ROI 指定が有効。

## RealityScan 側の手順

1. `fused.ply` を RS プロジェクトに **Import Model**
2. インポートしたモデルを選択して **Texture** を実行（撮影画像から単一モデルとして投影テクスチャ生成）
3. PLY が読めない場合は `--obj` を付けて `fused.obj` を使用（32M面級ではファイル巨大・時間注意）

## テスト

```bat
<env python> -m pytest tests -m "not slow"   # 単体 12件
<env python> -m pytest tests -m slow          # 擬似データ統合（GWメッシュに人工穴→復元検証）
```
