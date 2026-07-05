# MeshFusion

**Fuse a RealityScan High Detail mesh with a 3DGS-derived complement mesh into a single model — then texture it as one model in RealityScan.**

[日本語版 README はこちら / Japanese README](README.ja.md)

RealityScan can import external models, but it cannot merge multiple models into
one, and texturing always happens per model. MeshFusion solves this: it keeps
your RealityScan **High Detail mesh untouched as the primary surface**, cuts
**complement patches** out of a second mesh (e.g. one produced by
[Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping) from the same
photo set — great at thin structures and low-texture regions where
photogrammetry struggles), and writes a **single fused PLY** you can re-import
into RealityScan and texture as one model.

- **CPU-only core** — distance queries (embree BVH via Open3D), ICP alignment
  and patch selection all run on CPU. No GPU required for fusion.
- **Non-destructive** — input meshes are never modified; the RS mesh is never
  transformed (the complement mesh is registered onto it).
- **Semi-automated by design** — every run writes a report and optional
  previews (gray = RS, orange = added patches) so you can inspect and re-tune
  before committing.
- Verified on a 32M-face RealityScan export fused with a 6M-face Gaussian
  Wrapping mesh.

## Install

```bash
git clone https://github.com/toruhashimoto/rs-gw-mesh-fusion
cd rs-gw-mesh-fusion
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Python 3.10+ on Windows (the tool is developed and tested on Windows; the
Python core itself is platform-neutral).

## Quick start

**Desktop app (local Gradio UI):** double-click `launch_app.bat`, or:

```bash
python app.py
```

**CLI:**

```bash
run_fuse.bat --rs rs_high_detail.ply --gw complement.ply --out out\fusion --icp
```

Outputs in `--out`: `fused.ply` (single model, vertex colors kept),
`fusion_report.txt`, `fusion_meta.json`, `patch_faces.npy`, and with `--obj`
also `fused.obj`.

Then in RealityScan: **Import Model** → select the imported model → **Texture**.
RealityScan may warn about non-manifold edges (the fused model intentionally
contains separate shells); its built-in cleaning handles this, or pass
`--clean` to pre-remove degenerate/duplicate faces.

## Why `--icp` is (practically) required

**RealityScan's mesh export and its COLMAP export do not share a coordinate
frame** (we measured a small rotation about the origin — a median offset of
5.6 scene units on our test scene; scale is identical). Since a 3DGS
reconstruction lives in the COLMAP frame, the complement mesh must be
registered onto the RS mesh first. MeshFusion does this with a multi-scale ICP
(correspondence distance shrinking from the measured misalignment down to the
RS edge length, rigid first, scale-enabled only at the finest stages) and
reaches sub-edge-length accuracy. Without `--icp`, a frame mismatch is
detected and the run stops with a clear error instead of producing garbage.

## How it works

1. **Alignment check** — RS-vertex → complement-surface distance statistics;
   stops (or runs ICP with `--icp`) when the frames don't match.
2. **Patch selection** — complement faces farther than `tau` (default:
   8 × RS median edge length) from the RS surface, restricted to an ROI
   (default: RS bounding box + 10%; or a convex-hull JSON via `--roi_json`,
   compatible with Gaussian Wrapping's Blender bounding-volume add-on),
   with small confetti components removed and patch borders dilated a few
   rings so they overlap the RS surface (projection texturing is unaffected
   by overlaps).
3. **Fusion** — RS mesh + patches concatenated into one PLY. Topology stays
   disconnected on purpose: RealityScan textures by projection and does not
   need a watertight, connected mesh — just a single model.

## Tuning

| Symptom | Fix |
|---|---|
| Too much complement (room walls/floor get added) | Provide `--roi_json` around your object, or `--roi_expand 0` |
| Small fragments remain | Raise `--min_patch_area_ratio` (e.g. `1e-3`) |
| Holes in the RS mesh are not filled | Lower `--tau_factor` (e.g. `4`) |
| "Meshes do not appear to share a coordinate frame" | Add `--icp` (recommended always) |
| RealityScan non-manifold warnings | Add `--clean`, and/or accept RS's cleaning prompt |

## Full pipeline (RealityScan × Gaussian Wrapping)

See [docs/pipeline-guide.md](docs/pipeline-guide.md) for the end-to-end
workflow — RealityScan capture → COLMAP export → Gaussian Wrapping training &
mesh extraction (including the Windows / RTX 50-series build notes:
`NVCC_APPEND_FLAGS=-DUSE_CUDA` and friends) → fusion → texturing in
RealityScan.

## Tests

```bash
pytest tests -m "not slow"    # unit tests
pytest tests -m slow          # integration: cuts holes into a real mesh and
                              # verifies they are recovered as patches
```

## License

Non-commercial research use, following the terms of the Gaussian-Splatting
License — see [LICENSE.md](LICENSE.md). Gaussian Wrapping, gaussian-splatting
and nvdiffrast (optional preview dependency) are not bundled and carry their
own licenses.

## Acknowledgements

Built to complement [Gaussian Wrapping](https://github.com/diego1401/GaussianWrapping)
("From Blobs to Spokes: High-Fidelity Surface Reconstruction via Oriented
Gaussians", Gomez et al., 2026) and Epic Games' RealityScan. Uses
[trimesh](https://trimesh.org/), [Open3D](http://www.open3d.org/) and
optionally [nvdiffrast](https://github.com/NVlabs/nvdiffrast).
