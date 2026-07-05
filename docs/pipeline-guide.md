# End-to-end pipeline: RealityScan × Gaussian Wrapping × MeshFusion

Goal: photogrammetry (RealityScan High Detail) complemented by a 3DGS surface
reconstruction (Gaussian Wrapping) in regions where photogrammetry struggles
(thin structures, dark / low-texture / reflective surfaces), delivered as a
**single textured model** out of RealityScan.

```
RealityScan project
 ├─ (A) High Detail mesh  ──── export PLY ─────────────┐
 └─ (B) COLMAP export (images + sparse) ─┐             │
                                         ▼             ▼
                    (C) Gaussian Wrapping train+extract │
                                         │ mesh PLY    │
                                         ▼             ▼
                              (D) MeshFusion  --icp  ──► fused.ply
                                         │
                                         ▼
                    (E) RealityScan: Import Model → Texture (single model)
```

## (A)(B) RealityScan exports

- Reconstruct in RealityScan, export the **High Detail mesh** as PLY.
- Export the registration as **COLMAP format** (images + `sparse/` with
  `cameras.txt` / `images.txt` / `points3D.txt`). Undistorted images with a
  PINHOLE camera model are what downstream 3DGS code expects.
- **Important:** the mesh export and the COLMAP export do NOT share a
  coordinate frame (measured: a small rotation about the origin, several
  scene units of median offset; scale identical). This is why MeshFusion's
  `--icp` exists — do not try to fuse without it.

## (C) Gaussian Wrapping on Windows (RTX 50-series notes)

[Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping) officially
targets Linux + CUDA 11.8/12.1. The following adaptations are what we needed
for **Windows 11 + RTX 5070 Ti (Blackwell, sm_120) + CUDA 12.8 +
torch 2.9.1+cu128 + VS2022**; versions matter.

Key environment for building ALL CUDA extension submodules:

```bat
call vcvars64.bat
set CUDA_HOME=<CUDA 12.8 install>
set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
set VSLANG=1033
set NVCC_APPEND_FLAGS=-DUSE_CUDA   <-- the critical one
```

`NVCC_APPEND_FLAGS=-DUSE_CUDA` fixes the otherwise-fatal
`compiled_autograd.h: error C2872: 'std': ambiguous symbol` that every
torch-2.9 CUDA extension hits under MSVC (nvcc's cudafe re-emits
`::std::string` unqualified; torch already guards the code with
`#if defined(_WIN32) && defined(USE_CUDA)` but extension builds never define
`USE_CUDA`). With this one env var, all extensions — including fused-ssim,
long believed unbuildable on torch 2.9 + Windows — compile unmodified.

Source patches we needed (information only; apply to your own clone):

| File | Change | Why |
|---|---|---|
| `gaussian_wrapping/scene/dataset_readers.py` | fall back to `read_points3D_text` when `points3D.bin` is absent | RealityScan exports text-only COLMAP models |
| `gaussian_wrapping/texture_mesh.py` | `MeshRasterizer(..., use_opengl=False)` | OpenGL contexts are unreliable on Windows; the CUDA rasterizer is equivalent here |
| `gaussian_wrapping/train.py`, `arguments/__init__.py` | open `cfg_args` with `encoding="utf-8"` | locale-encoding round-trip breaks on non-ASCII paths |
| `gaussian_wrapping/pivot_based_mesh_extraction.py` | add `--delaunay_method {tetranerf,scipy}` | scipy fallback when the CGAL extension is unavailable |
| `submodules/diff-gaussian-rasterization_sof/.../__init__.py` | dataclass defaults → `field(default_factory=...)` (4 places) | Python 3.11 rejects mutable dataclass defaults |
| `submodules/nvdiffrast/nvdiffrast/torch/ops.py` | use `load()`'s return value instead of `importlib.import_module` | torch 2.9 pybind11 modules are no longer auto-registered in `sys.modules` |
| `submodules/tetra_triangulation/CMakeLists.txt` | pybind11 `v2.9.2 → v2.13.6`, guard `-flto=auto` behind `if(NOT MSVC)`, link `torch_python`, add CUDA toolkit include dirs | Python 3.11 + MSVC support |
| `submodules/tetra_triangulation/cmake/FindTorch.cmake` | guard GCC-only CXX flags behind `if(NOT MSVC)` | `cl` fails with D8021 |

Build `tetra_triangulation` with the **Ninja generator**
(`cmake -G Ninja . && cmake --build .`), not `make`: single-config output
avoids the VS `Release/` subdirectory problem. CGAL/GMP/MPFR come from
conda-forge (`cgal-cpp gmp mpfr`). Also `pip install matplotlib` (missing from
upstream requirements).

Then run end-to-end (about 60–80 min for a 74-image scene on a 5070 Ti):

```
python gaussian_wrapping/scripts/train_and_extract_gw_ours.py ^
    -s <COLMAP_DATASET> -m <OUT> --N_max_gaussians 2500000
```

`--N_max_gaussians 2500000` fits 16 GB VRAM (upstream default 6M targets
24 GB cards). The mesh to feed into MeshFusion is
`<OUT>/mesh_ours_2pivots_post.ply`.

## (D) Fusion

```
run_fuse.bat --rs <rs_high_detail.ply> --gw <OUT>/mesh_ours_2pivots_post.ply ^
             --out <FUSION_OUT> --icp
```

Read `fusion_report.txt`, look at the previews, and iterate on
`--tau_factor` / `--roi_json` / `--min_patch_area_ratio` (see README tuning
table). For object-centric captures, a Blender bounding volume around the
object (`--roi_json`, Gaussian Wrapping's add-on format) keeps room-scale GW
background out of the fusion.

## (E) Back into RealityScan

1. **Import Model** → `fused.ply` (use `--obj` output if your RS version
   rejects PLY).
2. RealityScan may warn about non-manifold edges — expected (the fused model
   contains separate shells); accept RS's cleaning (it only matters before
   texturing; UVs are regenerated anyway) or pre-run with `--clean`.
3. Select the imported model → **Texture**. Because it is now a single model,
   RealityScan textures it as one.
