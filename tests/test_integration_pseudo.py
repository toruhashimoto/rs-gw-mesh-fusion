import json
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fuse_meshes
import fusion_lib as fl
from make_pseudo_rs import make_pseudo

GW_REAL = r"D:\Claude\Photogrammetry\GW_Output\sample_ours\mesh_ours_2pivots_post.ply"


@pytest.mark.slow
def test_pseudo_recovery(tmp_path):
    full = trimesh.load(GW_REAL, process=False)
    gw = full.simplify_quadric_decimation(face_count=400_000)
    gw_path = str(tmp_path / "gw_dec.ply")
    gw.export(gw_path)

    meta = make_pseudo(gw_path, str(tmp_path), n_holes=3, hole_radius_factor=0.03, seed=1)
    removed = np.array(meta["removed_faces"])
    assert len(removed) > 500

    out = str(tmp_path / "fused")
    fuse_meshes.main(["--rs", str(tmp_path / "pseudo_rs.ply"), "--gw", gw_path,
                      "--out", out, "--seed", "1"])
    patch = np.load(os.path.join(out, "patch_faces.npy"))

    # recovery: most of the removed area must be re-added as patches
    area = gw.area_faces
    rec = area[np.intersect1d(patch, removed)].sum() / area[removed].sum()
    assert rec > 0.60, f"recovery too low: {rec:.2%}"

    # precision: patches outside removed+overlap must be tiny
    # (geometry is identical everywhere else)
    rm_mask = np.zeros(len(gw.faces), dtype=bool)
    rm_mask[removed] = True
    allowed = fl.dilate_faces(gw, rm_mask, rings=5)
    spurious = np.setdiff1d(patch, np.flatnonzero(allowed))
    assert area[spurious].sum() < 0.05 * area[removed].sum(), \
        f"spurious patches: {len(spurious)} faces, area {area[spurious].sum():.4g}"
