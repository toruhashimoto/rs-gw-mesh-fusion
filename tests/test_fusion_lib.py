import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_lib import (alignment_stats, build_raycast_scene, clean_mesh,
                        median_edge_length, unsigned_distance)


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


def test_clean_mesh_removes_degenerate_and_duplicate_faces():
    box = unit_box()
    v = np.vstack([box.vertices, [[9.0, 9.0, 9.0]]])          # + unreferenced vertex
    f = np.vstack([box.faces,
                   box.faces[0],                              # duplicate face
                   [0, 0, 1]])                                # degenerate face
    colors = np.full((len(v), 4), [10, 20, 30, 255], dtype=np.uint8)
    dirty = trimesh.Trimesh(v, f, vertex_colors=colors, process=False)

    cleaned, stats = clean_mesh(dirty)
    assert len(cleaned.faces) == len(box.faces)
    assert len(cleaned.vertices) == len(box.vertices)          # unreferenced dropped
    assert stats["faces_removed"] == 2
    assert stats["vertices_removed"] == 1
    vc = np.asarray(cleaned.visual.vertex_colors)
    assert vc.shape[0] == len(cleaned.vertices)                # colors stay in sync
    assert (vc[0][:3] == [10, 20, 30]).all()


def test_alignment_stats_detects_offset():
    m = unit_box()
    scene_aligned = build_raycast_scene(m.vertices, m.faces)
    shifted = unit_box()
    shifted.apply_translation([5.0, 0.0, 0.0])
    scene_shifted = build_raycast_scene(shifted.vertices, shifted.faces)
    ok = alignment_stats(m, scene_aligned, n_samples=100)
    bad = alignment_stats(m, scene_shifted, n_samples=100)
    assert ok["median"] < 1e-4
    assert bad["median"] > 2.0
