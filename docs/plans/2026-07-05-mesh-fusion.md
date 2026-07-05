# mesh_fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse a RealityScan High Detail mesh (primary) with a GaussianWrapping mesh (complement patches) into a single-file model for re-import & texturing in RealityScan.

**Architecture:** Pure-function core (`fusion_lib.py`) driven by a CLI (`fuse_meshes.py`). Distances via open3d RaycastingScene (embree BVH), mesh ops via trimesh. Pseudo-data generator + pytest for verification, nvdiffrast preview renderer, ASCII .bat launcher.

**Tech Stack:** Python 3.11 (conda env `gaussian_wrapping`), trimesh 4.6.8, open3d 0.19, numpy 2.x, scipy, nvdiffrast (preview only), pytest.

## Global Constraints

- Env python: `C:\Users\toruh\miniconda3\envs\gaussian_wrapping\python.exe` (below: `%PY%`)
- Tool root: `D:\Claude\Photogrammetry\MeshFusion\`
- Inputs are NEVER modified on disk (non-destructive principle)
- RS mesh is never transformed; ICP (opt-in) transforms the GW mesh only
- All text output files written with `encoding="utf-8"`
- .bat files: ASCII only, CRLF, no chcp (windows-launcher-scripts rules); run fixer after writing
- Defaults (from spec): `tau = rs_median_edge * 8`, ROI = RS AABB expanded 10% per axis,
  `min_patch_area_ratio = 1e-4` (of RS total area), `overlap_rings = 3`,
  alignment stop threshold = `rs_median_edge * 20`
- ROI JSON schema (GW Blender addon): `{"vertices": [[x,y,z], ...]}` world-space convex-hull
  vertices → containment test via `scipy.spatial.Delaunay(verts).find_simplex(p) >= 0`

---

### Task 1: Scaffold, git init, launcher

**Files:**
- Create: `D:\Claude\Photogrammetry\MeshFusion\{tests\,docs\}` (dirs), `.gitignore`, `run_fuse.bat`

**Interfaces:**
- Produces: repo skeleton; `run_fuse.bat` forwarding all args to `fuse_meshes.py` under the build/runtime env

- [ ] **Step 1:** `git init` in MeshFusion; `.gitignore` with `__pycache__/`, `*.pyc`, `output/`, `.pytest_cache/`
- [ ] **Step 2:** `%PY% -m pip install pytest` (only if `pytest` missing from env)
- [ ] **Step 3:** Write `run_fuse.bat` (ASCII): vcvars64 + CUDA_HOME=v12.8 + PATH(env Scripts/Library\bin) + `VSLANG=1033` + `NVCC_APPEND_FLAGS=-DUSE_CUDA` + `PYTHONUTF8=1`, then `"%PREFIX%\python.exe" "%~dp0fuse_meshes.py" %*` (same env block as `GaussianWrapping\run_gw.bat`)
- [ ] **Step 4:** Run encoding fixer on the .bat; verify `BOM=False CRLF=True nonASCII=False`
- [ ] **Step 5:** Commit "chore: scaffold MeshFusion"

### Task 2: fusion_lib core — distances, median edge, alignment

**Files:**
- Create: `D:\Claude\Photogrammetry\MeshFusion\fusion_lib.py`
- Test: `D:\Claude\Photogrammetry\MeshFusion\tests\test_fusion_lib.py`

**Interfaces (Produces):**
```python
build_raycast_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene
unsigned_distance(scene, points: np.ndarray, batch: int = 2_000_000) -> np.ndarray  # float32 [N]
median_edge_length(mesh: trimesh.Trimesh, sample: int = 200_000, seed: int = 0) -> float
alignment_stats(rs_mesh, gw_scene, n_samples: int = 100_000, seed: int = 0) -> dict
    # keys: "median", "p90", "p99", "max" (floats; RS-vertex → GW-surface distances)
```

- [ ] **Step 1: Failing tests** (`tests/test_fusion_lib.py`)

```python
import numpy as np, trimesh, pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_lib import build_raycast_scene, unsigned_distance, median_edge_length, alignment_stats

def unit_box():
    return trimesh.creation.box(extents=[2, 2, 2])  # spans [-1,1]^3

def test_unsigned_distance_on_box_surface_and_offset():
    m = unit_box()
    scene = build_raycast_scene(m.vertices, m.faces)
    d = unsigned_distance(scene, np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert d[0] == pytest.approx(0.0, abs=1e-5)
    assert d[1] == pytest.approx(2.0, abs=1e-4)

def test_median_edge_length_box():
    m = unit_box()
    assert median_edge_length(m) == pytest.approx(np.median(m.edges_unique_length), rel=1e-6)

def test_alignment_stats_detects_offset():
    m = unit_box()
    scene_aligned = build_raycast_scene(m.vertices, m.faces)
    shifted = unit_box(); shifted.apply_translation([5.0, 0.0, 0.0])
    scene_shifted = build_raycast_scene(shifted.vertices, shifted.faces)
    ok = alignment_stats(m, scene_aligned, n_samples=100)
    bad = alignment_stats(m, scene_shifted, n_samples=100)
    assert ok["median"] < 1e-4 and bad["median"] > 2.0
```

- [ ] **Step 2:** `%PY% -m pytest tests\test_fusion_lib.py -v` → FAIL (ModuleNotFoundError: fusion_lib)
- [ ] **Step 3: Implement** in `fusion_lib.py`:

```python
"""Core mesh-fusion primitives: BVH distances, edge stats, alignment check."""
import numpy as np
import open3d as o3d
import trimesh


def build_raycast_scene(vertices, faces):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(np.ascontiguousarray(vertices, dtype=np.float32)),
        o3d.core.Tensor(np.ascontiguousarray(faces, dtype=np.uint32)),
    )
    return scene


def unsigned_distance(scene, points, batch=2_000_000):
    pts = np.ascontiguousarray(points, dtype=np.float32)
    out = np.empty(len(pts), dtype=np.float32)
    for i in range(0, len(pts), batch):
        out[i:i + batch] = scene.compute_distance(o3d.core.Tensor(pts[i:i + batch])).numpy()
    return out


def median_edge_length(mesh, sample=200_000, seed=0):
    edges = mesh.edges_unique
    if len(edges) > sample:
        rng = np.random.default_rng(seed)
        edges = edges[rng.choice(len(edges), sample, replace=False)]
    v = mesh.vertices.view(np.ndarray)
    return float(np.median(np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)))


def alignment_stats(rs_mesh, gw_scene, n_samples=100_000, seed=0):
    rng = np.random.default_rng(seed)
    v = rs_mesh.vertices.view(np.ndarray)
    idx = rng.choice(len(v), min(n_samples, len(v)), replace=False)
    d = unsigned_distance(gw_scene, v[idx])
    return {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
            "p99": float(np.percentile(d, 99)), "max": float(d.max())}
```

- [ ] **Step 4:** `%PY% -m pytest tests\test_fusion_lib.py -v` → 3 passed
- [ ] **Step 5:** Commit "feat: fusion_lib distances + alignment stats"

### Task 3: patch selection — distance mask, ROI, components, dilation

**Files:**
- Modify: `fusion_lib.py` (append)
- Test: `tests\test_selection.py`

**Interfaces (Produces):**
```python
face_min_distance(gw_mesh, rs_scene, batch=2_000_000) -> np.ndarray  # float32 [F]
roi_mask_aabb(gw_mesh, bounds_min: np.ndarray, bounds_max: np.ndarray) -> np.ndarray  # bool [F]
roi_mask_hull(gw_mesh, hull_vertices: np.ndarray) -> np.ndarray  # bool [F]
filter_small_components(gw_mesh, face_mask, min_area: float) -> tuple[np.ndarray, int, int]
    # returns (new_mask, n_removed_components, n_kept_components)
dilate_faces(gw_mesh, face_mask, rings: int) -> np.ndarray
```

- [ ] **Step 1: Failing tests** (`tests/test_selection.py`)

```python
import numpy as np, trimesh, pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_lib import (build_raycast_scene, face_min_distance, roi_mask_aabb,
                        roi_mask_hull, filter_small_components, dilate_faces)

def rs_box():
    return trimesh.creation.box(extents=[2, 2, 2])

def gw_with_far_sphere():
    # identical box + sphere far above it (the "complement" geometry)
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    sph.apply_translation([0.0, 0.0, 3.0])
    return trimesh.util.concatenate([rs_box(), sph]), len(rs_box().faces)

def test_face_min_distance_selects_only_sphere():
    gw, n_box_faces = gw_with_far_sphere()
    scene = build_raycast_scene(rs_box().vertices, rs_box().faces)
    d = face_min_distance(gw, scene)
    tau = 0.5
    assert not (d[:n_box_faces] > tau).any()      # coincident box faces near zero
    assert (d[n_box_faces:] > tau).all()          # sphere fully beyond tau

def test_roi_masks():
    gw, n_box_faces = gw_with_far_sphere()
    inside = roi_mask_aabb(gw, np.array([-1.2, -1.2, -1.2]), np.array([1.2, 1.2, 1.2]))
    assert inside[:n_box_faces].all() and not inside[n_box_faces:].any()
    hull = np.array([[-2,-2,-2],[4,-2,-2],[-2,4,-2],[-2,-2,6],[4,4,6],[4,4,-2],[4,-2,6],[-2,4,6]], float)
    assert roi_mask_hull(gw, hull).all()

def test_filter_small_components_drops_confetti():
    big = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    tiny = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
    tiny.apply_translation([5, 0, 0])
    gw = trimesh.util.concatenate([big, tiny])
    mask = np.ones(len(gw.faces), dtype=bool)
    new_mask, removed, kept = filter_small_components(gw, mask, min_area=0.1)
    assert removed == 1 and kept == 1
    assert new_mask[:len(big.faces)].all() and not new_mask[len(big.faces):].any()

def test_dilate_faces_grows_selection():
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mask = np.zeros(len(m.faces), dtype=bool); mask[0] = True
    grown = dilate_faces(m, mask, rings=2)
    assert grown.sum() > 3 and grown[0]

def test_filter_small_components_empty_mask_ok():
    m = trimesh.creation.icosphere(subdivisions=1)
    mask = np.zeros(len(m.faces), dtype=bool)
    new_mask, removed, kept = filter_small_components(m, mask, min_area=0.1)
    assert not new_mask.any() and removed == 0 and kept == 0
```

- [ ] **Step 2:** Run → FAIL (ImportError: face_min_distance)
- [ ] **Step 3: Implement** (append to `fusion_lib.py`):

```python
def face_min_distance(gw_mesh, rs_scene, batch=2_000_000):
    v = gw_mesh.vertices.view(np.ndarray)
    f = gw_mesh.faces.view(np.ndarray)
    dv = unsigned_distance(rs_scene, v, batch)
    dc = unsigned_distance(rs_scene, v[f].mean(axis=1), batch)
    return np.minimum(np.minimum(dv[f[:, 0]], dv[f[:, 1]]),
                      np.minimum(dv[f[:, 2]], dc)).astype(np.float32)


def roi_mask_aabb(gw_mesh, bounds_min, bounds_max):
    c = gw_mesh.triangles_center
    return np.all((c >= bounds_min) & (c <= bounds_max), axis=1)


def roi_mask_hull(gw_mesh, hull_vertices):
    from scipy.spatial import Delaunay
    tri = Delaunay(np.asarray(hull_vertices, dtype=np.float64))
    return tri.find_simplex(gw_mesh.triangles_center) >= 0


def filter_small_components(gw_mesh, face_mask, min_area):
    from trimesh.graph import connected_components
    sel = np.flatnonzero(face_mask)
    if len(sel) == 0:
        return face_mask.copy(), 0, 0
    remap = np.full(len(gw_mesh.faces), -1, dtype=np.int64)
    remap[sel] = np.arange(len(sel))
    adj = gw_mesh.face_adjacency
    both = face_mask[adj[:, 0]] & face_mask[adj[:, 1]]
    comps = connected_components(remap[adj[both]], nodes=np.arange(len(sel)))
    areas = gw_mesh.area_faces[sel]
    out = face_mask.copy()
    removed = kept = 0
    for comp in comps:
        if areas[comp].sum() < min_area:
            out[sel[comp]] = False
            removed += 1
        else:
            kept += 1
    return out, removed, kept


def dilate_faces(gw_mesh, face_mask, rings):
    adj = gw_mesh.face_adjacency
    mask = face_mask.copy()
    for _ in range(rings):
        m0, m1 = mask[adj[:, 0]], mask[adj[:, 1]]
        grow0, grow1 = adj[:, 0][m1 & ~m0], adj[:, 1][m0 & ~m1]
        if len(grow0) == 0 and len(grow1) == 0:
            break
        mask[grow0] = True
        mask[grow1] = True
    return mask
```

- [ ] **Step 4:** Run → 5 passed (plus Task 2's 3 still passing)
- [ ] **Step 5:** Commit "feat: patch selection (distance/ROI/components/dilation)"

### Task 4: fuse + report + CLI

**Files:**
- Modify: `fusion_lib.py` (append `fuse_meshes` function)
- Create: `fuse_meshes.py` (CLI)
- Test: `tests\test_cli.py`

**Interfaces (Produces):**
```python
# fusion_lib
fuse(rs_mesh, gw_mesh, face_mask) -> tuple[trimesh.Trimesh, int, float]  # (fused, n_patch_faces, patch_area)
# fuse_meshes.py
main(argv: list[str] | None = None) -> dict   # stats dict; also writes files to --out
# CLI: --rs --gw --out [--tau F] [--tau_factor 8] [--roi_expand 0.10] [--roi_json P]
#      [--min_patch_area_ratio 1e-4] [--overlap_rings 3] [--icp] [--obj]
#      [--align_factor 20] [--seed 0]
# writes: fused.ply, patch_faces.npy (int64 GW face indices), fusion_report.txt
# exit codes: 0 ok (incl. zero patches), 2 alignment failure
```

- [ ] **Step 1: Failing test** (`tests/test_cli.py`)

```python
import numpy as np, trimesh, sys, os, subprocess, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuse_meshes

def make_inputs(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    sph.apply_translation([0.0, 0.0, 1.05])          # sits just above the box top => inside ROI
    gw = trimesh.util.concatenate([rs.copy(), sph])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p); gw.export(gw_p)
    return rs_p, gw_p, len(rs.faces), len(sph.faces)

def test_cli_end_to_end(tmp_path):
    rs_p, gw_p, n_rs, n_sph = make_inputs(tmp_path)
    out = str(tmp_path / "out")
    stats = fuse_meshes.main(["--rs", rs_p, "--gw", gw_p, "--out", out,
                              "--tau_factor", "1.0", "--overlap_rings", "1"])
    fused = trimesh.load(os.path.join(out, "fused.ply"), process=False)
    assert len(fused.faces) > n_rs                       # patches were added
    assert stats["n_patch_faces"] > 0
    assert os.path.exists(os.path.join(out, "fusion_report.txt"))
    idx = np.load(os.path.join(out, "patch_faces.npy"))
    assert (idx >= n_rs).sum() > 0.8 * len(idx)          # patches come from the sphere part

def test_cli_zero_patches_when_identical(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p); rs.export(gw_p)
    stats = fuse_meshes.main(["--rs", rs_p, "--gw", gw_p, "--out", str(tmp_path / "o")])
    assert stats["n_patch_faces"] == 0                   # normal exit, explicit zero

def test_cli_alignment_failure_exits_2(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    gw = trimesh.creation.box(extents=[2, 2, 2]); gw.apply_translation([50, 0, 0])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p); gw.export(gw_p)
    try:
        fuse_meshes.main(["--rs", rs_p, "--gw", gw_p, "--out", str(tmp_path / "o")])
        assert False, "should have raised SystemExit(2)"
    except SystemExit as e:
        assert e.code == 2
```

- [ ] **Step 2:** Run → FAIL (no module fuse_meshes)
- [ ] **Step 3: Implement.** `fusion_lib.fuse`:

```python
def _vertex_colors_or_gray(mesh):
    try:
        vc = np.asarray(mesh.visual.vertex_colors)
        if vc.ndim == 2 and vc.shape[0] == len(mesh.vertices):
            return vc[:, :4].astype(np.uint8)
    except Exception:
        pass
    return np.full((len(mesh.vertices), 4), [200, 200, 200, 255], dtype=np.uint8)


def fuse(rs_mesh, gw_mesh, face_mask):
    rs_out = trimesh.Trimesh(rs_mesh.vertices, rs_mesh.faces,
                             vertex_colors=_vertex_colors_or_gray(rs_mesh), process=False)
    if not face_mask.any():
        return rs_out, 0, 0.0
    patch = gw_mesh.submesh([np.flatnonzero(face_mask)], append=True)
    patch = trimesh.Trimesh(patch.vertices, patch.faces,
                            vertex_colors=_vertex_colors_or_gray(patch), process=False)
    fused = trimesh.util.concatenate([rs_out, patch])
    return fused, len(patch.faces), float(patch.area)
```

`fuse_meshes.py` (complete):

```python
"""Fuse a RealityScan High Detail mesh with GaussianWrapping complement patches.

RS mesh is primary and never modified; GW faces farther than tau from the RS
surface (inside the ROI) are added as patches. Output: single fused.ply for
re-import & texturing in RealityScan. See docs/specs for the design.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh

import fusion_lib as fl


def parse_args(argv):
    p = argparse.ArgumentParser(description="RS-primary + GW-complement mesh fusion")
    p.add_argument("--rs", required=True, help="RealityScan High Detail mesh (PLY/OBJ)")
    p.add_argument("--gw", required=True, help="GaussianWrapping mesh (PLY)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--tau", type=float, default=None, help="absolute distance threshold")
    p.add_argument("--tau_factor", type=float, default=8.0, help="tau = rs_median_edge * factor")
    p.add_argument("--roi_expand", type=float, default=0.10, help="RS AABB expansion ratio")
    p.add_argument("--roi_json", default=None, help="GW Blender bounding-volume JSON (convex hull)")
    p.add_argument("--min_patch_area_ratio", type=float, default=1e-4)
    p.add_argument("--overlap_rings", type=int, default=3)
    p.add_argument("--icp", action="store_true", help="allow scaled ICP refinement of the GW mesh")
    p.add_argument("--obj", action="store_true", help="also export fused.obj (large!)")
    p.add_argument("--align_factor", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def icp_refine(rs_mesh, gw_mesh, max_corr, n_sample=500_000, seed=0):
    import open3d as o3d
    rng = np.random.default_rng(seed)

    def sample_pcd(mesh):
        v = mesh.vertices.view(np.ndarray)
        idx = rng.choice(len(v), min(n_sample, len(v)), replace=False)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v[idx])
        return pcd

    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    res = o3d.pipelines.registration.registration_icp(
        sample_pcd(gw_mesh), sample_pcd(rs_mesh), max_corr, np.eye(4), est)
    return np.asarray(res.transformation), float(res.fitness), float(res.inlier_rmse)


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    log(f"[INFO] RS mesh : {args.rs}")
    log(f"[INFO] GW mesh : {args.gw}")
    rs = trimesh.load(args.rs, process=False)
    gw = trimesh.load(args.gw, process=False)
    log(f"[INFO] RS: V={len(rs.vertices):,} F={len(rs.faces):,}")
    log(f"[INFO] GW: V={len(gw.vertices):,} F={len(gw.faces):,}")

    rs_med_edge = fl.median_edge_length(rs, seed=args.seed)
    log(f"[INFO] RS median edge length = {rs_med_edge:.6g}")

    # [1] alignment check (RS vertices -> GW surface)
    gw_scene = fl.build_raycast_scene(gw.vertices, gw.faces)
    st = fl.alignment_stats(rs, gw_scene, seed=args.seed)
    align_thresh = args.align_factor * rs_med_edge
    log(f"[INFO] alignment RS->GW: median={st['median']:.6g} p90={st['p90']:.6g} "
        f"p99={st['p99']:.6g} (threshold {align_thresh:.6g})")
    if st["median"] > align_thresh and args.icp:
        max_corr = 50.0 * rs_med_edge
        T, fitness, rmse = icp_refine(rs, gw, max_corr, seed=args.seed)
        gw.apply_transform(T)
        log(f"[INFO] ICP applied (fitness={fitness:.3f} rmse={rmse:.6g}); rechecking")
        gw_scene = fl.build_raycast_scene(gw.vertices, gw.faces)
        st = fl.alignment_stats(rs, gw_scene, seed=args.seed)
        log(f"[INFO] alignment after ICP: median={st['median']:.6g}")
    if st["median"] > align_thresh:
        log("[ERROR] Meshes do not appear to share a coordinate frame. "
            "Verify the RS export settings, or re-run with --icp.")
        report_path = os.path.join(args.out, "fusion_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        sys.exit(2)

    # [2] patch selection
    tau = args.tau if args.tau is not None else args.tau_factor * rs_med_edge
    log(f"[INFO] tau = {tau:.6g}")
    rs_scene = fl.build_raycast_scene(rs.vertices, rs.faces)
    dmin = fl.face_min_distance(gw, rs_scene)
    mask = dmin > tau
    log(f"[INFO] faces beyond tau: {int(mask.sum()):,} / {len(mask):,}")

    if args.roi_json:
        with open(args.roi_json, encoding="utf-8") as f:
            hull_vertices = np.asarray(json.load(f)["vertices"], dtype=np.float64)
        roi = fl.roi_mask_hull(gw, hull_vertices)
        log(f"[INFO] ROI (convex hull from {args.roi_json}): {int(roi.sum()):,} faces inside")
    else:
        span = rs.bounds[1] - rs.bounds[0]
        lo = rs.bounds[0] - args.roi_expand * span
        hi = rs.bounds[1] + args.roi_expand * span
        roi = fl.roi_mask_aabb(gw, lo, hi)
        log(f"[INFO] ROI (RS AABB +{args.roi_expand:.0%}): {int(roi.sum()):,} faces inside")
    mask &= roi
    log(f"[INFO] after ROI: {int(mask.sum()):,} faces")

    min_area = args.min_patch_area_ratio * rs.area
    mask, removed, kept = fl.filter_small_components(gw, mask, min_area)
    log(f"[INFO] components: kept {kept}, removed {removed} (< area {min_area:.6g})")

    mask = fl.dilate_faces(gw, mask, args.overlap_rings) & roi
    log(f"[INFO] after {args.overlap_rings}-ring overlap dilation: {int(mask.sum()):,} faces")

    # [3] fuse + outputs
    fused, n_patch, patch_area = fl.fuse(rs, gw, mask)
    if n_patch == 0:
        log("[INFO] No complement patches selected - RS mesh appears complete. "
            "Output equals the RS mesh (with vertex colors normalized).")
    log(f"[INFO] patches: {n_patch:,} faces, area {patch_area:.6g} "
        f"({patch_area / rs.area:.2%} of RS area)")
    ply_path = os.path.join(args.out, "fused.ply")
    fused.export(ply_path)
    log(f"[INFO] wrote {ply_path} (V={len(fused.vertices):,} F={len(fused.faces):,})")
    np.save(os.path.join(args.out, "patch_faces.npy"), np.flatnonzero(mask))
    if args.obj:
        obj_path = os.path.join(args.out, "fused.obj")
        fused.export(obj_path)
        log(f"[INFO] wrote {obj_path}")

    with open(os.path.join(args.out, "fusion_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return {"n_patch_faces": n_patch, "patch_area": patch_area,
            "alignment": st, "tau": tau, "components_kept": kept, "components_removed": removed}


if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** `%PY% -m pytest tests -v` → all pass (3+5+3)
- [ ] **Step 5:** Commit "feat: fuse + report + CLI"

### Task 5: pseudo-RS generator + slow integration self-test

**Files:**
- Create: `make_pseudo_rs.py`
- Test: `tests\test_integration_pseudo.py` (`@pytest.mark.slow`)

**Interfaces (Produces):**
```python
# make_pseudo_rs.py
make_pseudo(gw_path, out_dir, n_holes=3, hole_radius_factor=0.03, seed=0) -> dict
# writes pseudo_rs.ply + holes.json {"centers": [[xyz]..], "radius": r, "removed_faces": [int..]}
# holes are cut where GW faces are DENSE (central object), radius = ||extents|| * factor
```

- [ ] **Step 1:** Write `make_pseudo_rs.py`:

```python
"""Cut spherical holes into a GW mesh to fabricate a pseudo-RS mesh for self-testing."""
import argparse
import json
import os

import numpy as np
import trimesh


def make_pseudo(gw_path, out_dir, n_holes=3, hole_radius_factor=0.03, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    gw = trimesh.load(gw_path, process=False)
    rng = np.random.default_rng(seed)
    c = gw.triangles_center
    # pick hole centers in the densest region (central object, not background):
    # sample candidate faces near the median center of all faces
    center = np.median(c, axis=0)
    r_sel = np.linalg.norm(gw.extents) * 0.15
    cand = np.flatnonzero(np.linalg.norm(c - center, axis=1) < r_sel)
    if len(cand) < n_holes:
        cand = np.arange(len(c))
    centers = c[rng.choice(cand, n_holes, replace=False)]
    radius = float(np.linalg.norm(gw.extents) * hole_radius_factor)
    removed = np.zeros(len(gw.faces), dtype=bool)
    for ctr in centers:
        removed |= np.linalg.norm(c - ctr, axis=1) < radius
    pseudo = gw.submesh([np.flatnonzero(~removed)], append=True)
    pseudo_path = os.path.join(out_dir, "pseudo_rs.ply")
    pseudo.export(pseudo_path)
    meta = {"centers": centers.tolist(), "radius": radius,
            "removed_faces": np.flatnonzero(removed).tolist()}
    with open(os.path.join(out_dir, "holes.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"[INFO] pseudo RS: removed {int(removed.sum()):,} faces "
          f"in {n_holes} holes (r={radius:.4g}) -> {pseudo_path}")
    return meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_holes", type=int, default=3)
    p.add_argument("--hole_radius_factor", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    make_pseudo(a.gw, a.out, a.n_holes, a.hole_radius_factor, a.seed)
```

- [ ] **Step 2: Integration test** (`tests/test_integration_pseudo.py`) — uses a *decimated* GW mesh for speed:

```python
import json
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuse_meshes
from make_pseudo_rs import make_pseudo

GW_REAL = r"D:\Claude\Photogrammetry\GW_Output\sample_ours\mesh_ours_2pivots_post.ply"


@pytest.mark.slow
def test_pseudo_recovery(tmp_path):
    full = trimesh.load(GW_REAL, process=False)
    gw, _ = full.simplify_quadric_decimation(face_count=400_000), None
    gw_path = str(tmp_path / "gw_dec.ply")
    gw.export(gw_path)

    meta = make_pseudo(gw_path, str(tmp_path), n_holes=3, hole_radius_factor=0.03, seed=1)
    removed = np.array(meta["removed_faces"])
    assert len(removed) > 500

    out = str(tmp_path / "fused")
    stats = fuse_meshes.main(["--rs", str(tmp_path / "pseudo_rs.ply"), "--gw", gw_path,
                              "--out", out, "--seed", "1"])
    patch = np.load(os.path.join(out, "patch_faces.npy"))

    # recovery: most of the removed area must be re-added as patches
    area = gw.area_faces
    rec = area[np.intersect1d(patch, removed)].sum() / area[removed].sum()
    assert rec > 0.60, f"recovery too low: {rec:.2%}"

    # precision: patches outside removed+overlap must be tiny (identical geometry elsewhere)
    import fusion_lib as fl
    rm_mask = np.zeros(len(gw.faces), dtype=bool)
    rm_mask[removed] = True
    allowed = fl.dilate_faces(gw, rm_mask, rings=5)
    spurious = np.setdiff1d(patch, np.flatnonzero(allowed))
    assert area[spurious].sum() < 0.05 * area[removed].sum(), \
        f"spurious patches: {len(spurious)} faces"
```

- [ ] **Step 3:** Register the `slow` marker in `pytest.ini` (`markers = slow: heavy integration tests`); run `%PY% -m pytest tests -m slow -v` → 1 passed (allow several minutes). If `simplify_quadric_decimation` needs `fast_simplification`, install it (`%PY% -m pip install fast-simplification`).
- [ ] **Step 4:** `%PY% -m pytest tests -m "not slow" -v` → all fast tests still pass
- [ ] **Step 5:** Commit "feat: pseudo-RS generator + integration self-test"

### Task 6: preview renderer

**Files:**
- Create: `render_compare.py`

**Interfaces:**
- Consumes: `fused` outputs (`--rs`, `--gw`, `--patch_npy`), writes `preview_view1.png`, `preview_view2.png` to `--out`
- RS part painted gray RGB(180,180,180); patch part orange RGB(255,140,0)

- [ ] **Step 1:** Write `render_compare.py` reusing the camera/rasterize pattern from the session's `render_preview.py`: load RS mesh + GW mesh + `patch_faces.npy`; build one combined vertex/face/color buffer (RS gray, GW patch faces orange); look-at cameras from combined AABB (two azimuths 0.35 / 2.45 rad, fovy 55°, eye at 0.40 × extent-norm); `dr.RasterizeCudaContext` + `dr.interpolate`; white background; save with imageio. (Complete camera math is identical to `scratchpad/render_preview.py` — copy those `look_at` / `perspective` helpers verbatim.)
- [ ] **Step 2:** Smoke-run on the Task 4 CLI test outputs (box+sphere): both PNGs exist, nonzero coverage, orange pixels present (assert via `(img[...,0]>200)&(img[...,1]>100)&(img[...,2]<80)` count > 0 in a `if __name__` self-check or quick inline run).
- [ ] **Step 3:** Commit "feat: fusion preview renderer"

### Task 7: real-data run + verification (Sample_RS-ply × GW sample)

**Files:**
- Uses: `D:\Claude\Photogrammetry\Sample_RS-ply\Project_____1.ply`, `D:\Claude\Photogrammetry\GW_Output\sample_ours\mesh_ours_2pivots_post.ply`
- Output: `D:\Claude\Photogrammetry\GW_Output\fusion_sample\`

- [ ] **Step 1:** `run_fuse.bat` equivalent: `fuse_meshes.py --rs ...Project_____1.ply --gw ...mesh_ours_2pivots_post.ply --out ...fusion_sample` (default params, run in background; expect BVH build on 32M faces ≈ minutes, total ≲ 15 min)
- [ ] **Step 2:** Read `fusion_report.txt`: check alignment median ≪ threshold (frames should coincide); patch count/area plausible; components removed count
- [ ] **Step 3:** `render_compare.py` on the outputs → visually inspect patches (orange = GW complement regions)
- [ ] **Step 4:** If alignment fails: STOP and report to user (RS export coordinate settings) — do NOT silently ICP
- [ ] **Step 5:** Send previews + report summary to user; hand off `fused.ply` for RealityScan import + texture test
- [ ] **Step 6:** Commit "docs: real-data fusion run results" (report copy into docs/)

## Self-Review Notes

- Spec coverage: alignment check §3→Task 2/4; patch logic §2→Task 3; outputs §4→Task 4/6; pseudo-test §6-1→Task 5; real-data §6-2→Task 7; launcher §5→Task 1. OBJ export is opt-in `--obj` (32M-face OBJ is impractically large as a default — deviation from spec noted, PLY is the primary format).
- Type consistency: `face_mask` is always `np.ndarray[bool]` over GW faces; `patch_faces.npy` stores `np.flatnonzero(mask)` int64 indices; `main()` returns the stats dict used by tests.
- No placeholders: all code inline above.
