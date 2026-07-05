import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_lib import (build_raycast_scene, dilate_faces, face_min_distance,
                        filter_small_components, roi_mask_aabb, roi_mask_hull)


def rs_box():
    return trimesh.creation.box(extents=[2, 2, 2])


def gw_with_far_sphere():
    # identical box + sphere far above it (the "complement" geometry)
    sph = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    sph.apply_translation([0.0, 0.0, 3.0])
    return trimesh.util.concatenate([rs_box(), sph]), len(rs_box().faces)


def test_face_min_distance_selects_only_sphere():
    gw, n_box_faces = gw_with_far_sphere()
    box = rs_box()
    scene = build_raycast_scene(box.vertices, box.faces)
    d = face_min_distance(gw, scene)
    tau = 0.5
    assert not (d[:n_box_faces] > tau).any()      # coincident box faces are near zero
    assert (d[n_box_faces:] > tau).all()          # sphere is fully beyond tau


def test_roi_masks():
    gw, n_box_faces = gw_with_far_sphere()
    inside = roi_mask_aabb(gw, np.array([-1.2, -1.2, -1.2]), np.array([1.2, 1.2, 1.2]))
    assert inside[:n_box_faces].all()
    assert not inside[n_box_faces:].any()
    hull = np.array([[-2, -2, -2], [4, -2, -2], [-2, 4, -2], [-2, -2, 6],
                     [4, 4, 6], [4, 4, -2], [4, -2, 6], [-2, 4, 6]], dtype=float)
    assert roi_mask_hull(gw, hull).all()


def test_filter_small_components_drops_confetti():
    big = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    tiny = trimesh.creation.icosphere(subdivisions=0, radius=0.01)
    tiny.apply_translation([5, 0, 0])
    gw = trimesh.util.concatenate([big, tiny])
    mask = np.ones(len(gw.faces), dtype=bool)
    new_mask, removed, kept = filter_small_components(gw, mask, min_area=0.1)
    assert removed == 1 and kept == 1
    assert new_mask[:len(big.faces)].all()
    assert not new_mask[len(big.faces):].any()


def test_dilate_faces_grows_selection():
    m = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    mask = np.zeros(len(m.faces), dtype=bool)
    mask[0] = True
    grown = dilate_faces(m, mask, rings=2)
    assert grown.sum() > 3
    assert grown[0]


def test_filter_small_components_empty_mask_ok():
    m = trimesh.creation.icosphere(subdivisions=1)
    mask = np.zeros(len(m.faces), dtype=bool)
    new_mask, removed, kept = filter_small_components(m, mask, min_area=0.1)
    assert not new_mask.any()
    assert removed == 0 and kept == 0
