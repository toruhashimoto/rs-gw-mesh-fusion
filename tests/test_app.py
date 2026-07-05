import os
import sys

import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app


def test_run_fusion_generator_end_to_end(tmp_path):
    rs = trimesh.creation.box(extents=[2, 2, 2])
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.3)
    sph.apply_translation([0.0, 0.0, 1.5])
    gw = trimesh.util.concatenate([rs.copy(), sph])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p)
    gw.export(gw_p)
    out = str(tmp_path / "out")

    results = list(app.run_fusion(
        rs_p, gw_p, out, "", tau_factor=8, tau_abs=0.05, roi_expand=0.5,
        min_patch_area_ratio=1e-4, overlap_rings=1,
        use_icp=True, do_clean=True, export_obj=False, make_preview=False))
    log, report, img1, img2 = results[-1]
    assert "[DONE]" in log
    assert "patches:" in report            # report loaded and non-trivial
    assert os.path.isfile(os.path.join(out, "fused.ply"))


def test_run_fusion_none_optionals(tmp_path):
    # regression class: Gradio passes None for untouched/cleared fields
    rs = trimesh.creation.box(extents=[2, 2, 2])
    rs_p, gw_p = str(tmp_path / "rs.ply"), str(tmp_path / "gw.ply")
    rs.export(rs_p)
    rs.export(gw_p)
    results = list(app.run_fusion(
        rs_p, gw_p, str(tmp_path / "o"), None,
        tau_factor=None, tau_abs=None, roi_expand=None,
        min_patch_area_ratio=None, overlap_rings=None,
        use_icp=True, do_clean=False, export_obj=False, make_preview=False))
    log = results[-1][0]
    assert "[DONE]" in log and "None" not in log.split("fuse_meshes.py")[-1].split("\n")[0]


def test_run_fusion_rejects_missing_input(tmp_path):
    results = list(app.run_fusion(
        str(tmp_path / "nope.ply"), str(tmp_path / "nope2.ply"), str(tmp_path / "o"),
        "", 8, 0, 0.1, 1e-4, 3, True, False, False, False))
    log = results[-1][0]
    assert "[ERROR]" in log
