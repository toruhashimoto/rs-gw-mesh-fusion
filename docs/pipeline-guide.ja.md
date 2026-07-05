# パイプライン全体ガイド: RealityScan × Gaussian Wrapping × MeshFusion

目的: フォトグラメトリ（RealityScan High Detail）が苦手な領域（薄物・暗部・
低テクスチャ・反射面）を 3DGS 表面再構成（Gaussian Wrapping）で補完し、
**単一のテクスチャ付きモデル**として RealityScan から出力する。

```
RealityScan プロジェクト
 ├─ (A) High Detail メッシュ ── PLY エクスポート ───────┐
 └─ (B) COLMAP エクスポート (画像 + sparse) ─┐          │
                                            ▼          ▼
                   (C) Gaussian Wrapping 学習+メッシュ抽出
                                            │ メッシュ PLY
                                            ▼          ▼
                              (D) MeshFusion --icp ──► fused.ply
                                            │
                                            ▼
                   (E) RealityScan: Import Model → Texture（単一モデル）
```

## (A)(B) RealityScan からのエクスポート

- High Detail メッシュを PLY でエクスポート
- レジストレーションを **COLMAP 形式**でエクスポート（画像 + `sparse/` の
  `cameras.txt` / `images.txt` / `points3D.txt`）。下流の 3DGS コードは
  歪み補正済み画像 + PINHOLE モデルを前提とする
- **重要:** メッシュエクスポートと COLMAP エクスポートは**座標系が一致しない**
  （実測: 原点周りの微小回転、中央値で数シーンユニットのズレ。スケールは一致）。
  MeshFusion の `--icp` はこのためにある。`--icp` なしでの融合は不可

## (C) Gaussian Wrapping の Windows ビルド（RTX 50 系の知見）

[Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping) の公式対応は
Linux + CUDA 11.8/12.1。以下は **Windows 11 + RTX 5070 Ti (Blackwell, sm_120) +
CUDA 12.8 + torch 2.9.1+cu128 + VS2022** で動かすための実証済み適応。

全 CUDA 拡張のビルドに必要な環境変数:

```bat
call vcvars64.bat
set CUDA_HOME=<CUDA 12.8>
set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
set VSLANG=1033
set NVCC_APPEND_FLAGS=-DUSE_CUDA   ← 最重要
```

`NVCC_APPEND_FLAGS=-DUSE_CUDA` は、torch 2.9 + MSVC で全 CUDA 拡張が踏む
`compiled_autograd.h: error C2872: 'std': あいまいなシンボル` の正攻法の修正。
（nvcc の cudafe が `::std::string` を修飾なしで再出力するのが原因。torch 側には
`#if defined(_WIN32) && defined(USE_CUDA)` のガードが既にあるが、拡張ビルドでは
USE_CUDA が定義されないため効かない。）この1変数で、「torch 2.9 + Windows では
ビルド不能」とされてきた fused-ssim を含む全拡張が無改変でコンパイルできる。

必要だったソースパッチ（情報として記載。各自のクローンに適用）:

| ファイル | 変更 | 理由 |
|---|---|---|
| `gaussian_wrapping/scene/dataset_readers.py` | `points3D.bin` が無ければ `read_points3D_text` にフォールバック | RealityScan の COLMAP 出力は text のみ |
| `gaussian_wrapping/texture_mesh.py` | `MeshRasterizer(..., use_opengl=False)` | Windows の OpenGL コンテキストは不安定。CUDA ラスタライザで等価 |
| `gaussian_wrapping/train.py`, `arguments/__init__.py` | `cfg_args` を `encoding="utf-8"` で開く | cp932 環境の非 ASCII パスで破綻するため |
| `gaussian_wrapping/pivot_based_mesh_extraction.py` | `--delaunay_method {tetranerf,scipy}` を追加 | CGAL 拡張が無い環境向けの scipy フォールバック |
| `submodules/diff-gaussian-rasterization_sof/.../__init__.py` | dataclass 既定値を `field(default_factory=...)` に（4箇所） | Python 3.11 は可変デフォルトを拒否 |
| `submodules/nvdiffrast/nvdiffrast/torch/ops.py` | `importlib.import_module` ではなく `load()` の戻り値を使用 | torch 2.9 の pybind11 モジュールは sys.modules に自動登録されない |
| `submodules/tetra_triangulation/CMakeLists.txt` | pybind11 `v2.9.2 → v2.13.6`、`-flto=auto` を `if(NOT MSVC)` で除外、`torch_python` をリンク、CUDA include 追加 | Python 3.11 + MSVC 対応 |
| `submodules/tetra_triangulation/cmake/FindTorch.cmake` | GCC 専用 CXX フラグを `if(NOT MSVC)` で除外 | cl が D8021 で失敗 |

`tetra_triangulation` は `make` ではなく **Ninja**（`cmake -G Ninja . && cmake --build .`）
でビルド（single-config なので VS の `Release/` サブディレクトリ問題を回避）。
CGAL/GMP/MPFR は conda-forge（`cgal-cpp gmp mpfr`）。upstream の requirements に
無い `matplotlib` も `pip install` が必要。

実行（74枚シーン、RTX 5070 Ti で約60〜80分）:

```
python gaussian_wrapping/scripts/train_and_extract_gw_ours.py ^
    -s <COLMAPデータセット> -m <OUT> --N_max_gaussians 2500000
```

`--N_max_gaussians 2500000` は 16GB VRAM 向け（公式既定 6M は 24GB 級向け）。
MeshFusion に渡すのは `<OUT>/mesh_ours_2pivots_post.ply`。

## (D) 融合

```
run_fuse.bat --rs <rs_high_detail.ply> --gw <OUT>/mesh_ours_2pivots_post.ply ^
             --out <FUSION_OUT> --icp
```

`fusion_report.txt` とプレビューを確認し、`--tau_factor` / `--roi_json` /
`--min_patch_area_ratio` を調整して再実行（README のチューニング表参照）。
対象物中心の撮影では、Blender で対象物を囲む箱を作って `--roi_json` 指定
（Gaussian Wrapping の Blender アドオン形式）すると、部屋スケールの GW 背景の
混入を防げる。

## (E) RealityScan へ戻す

1. **Import Model** → `fused.ply`（PLY が読めない RS バージョンは `--obj` 出力を使用）
2. 非マニホールドエッジの警告は想定内（融合モデルは複数シェルを含む）。
   RS のクリーニングを実行してよい（テクスチャ前なので UV 消失は無関係）。
   `--clean` での事前軽減も可
3. インポートしたモデルを選択 → **Texture**。単一モデルなので一体でテクスチャが生成される
