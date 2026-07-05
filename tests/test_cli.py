import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuse_meshes


def make_inputs(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    # detached sphere above the box: min distance to box top = 0.2
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    sph.apply_translation([0.0, 0.0, 1.5])
    gw = trimesh.util.concatenate([rs.copy(), sph])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p)
    gw.export(gw_p)
    return rs_p, gw_p, len(rs.faces), len(sph.faces)


def test_cli_end_to_end(tmp_path):
    rs_p, gw_p, n_rs, n_sph = make_inputs(tmp_path)
    out = str(tmp_path / "out")
    # NOTE: a 12-face box has median edge 2.0, so tau_factor-based thresholds are
    # meaningless here - use an absolute --tau. ROI must be widened to reach z=1.8.
    stats = fuse_meshes.main(["--rs", rs_p, "--gw", gw_p, "--out", out,
                              "--tau", "0.05", "--roi_expand", "0.5",
                              "--overlap_rings", "1"])
    fused = trimesh.load(os.path.join(out, "fused.ply"), process=False)
    assert len(fused.faces) > n_rs                       # patches were added
    assert stats["n_patch_faces"] > 0
    assert os.path.exists(os.path.join(out, "fusion_report.txt"))
    idx = np.load(os.path.join(out, "patch_faces.npy"))
    assert (idx >= n_rs).sum() > 0.8 * len(idx)          # patches come from the sphere part


def test_cli_zero_patches_when_identical(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p)
    rs.export(gw_p)
    stats = fuse_meshes.main(["--rs", rs_p, "--gw", gw_p, "--out", str(tmp_path / "o")])
    assert stats["n_patch_faces"] == 0                   # normal exit, explicit zero


def test_cli_alignment_failure_exits_2(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    gw = trimesh.creation.box(extents=[2, 2, 2])
    gw.apply_translation([50, 0, 0])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p)
    gw.export(gw_p)
    try:
        fuse_meshes.main(["--rs", rs_p, "--gw", gw_p, "--out", str(tmp_path / "o")])
        assert False, "should have raised SystemExit(2)"
    except SystemExit as e:
        assert e.code == 2
