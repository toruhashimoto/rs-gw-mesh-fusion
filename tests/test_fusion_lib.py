import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion_lib import (alignment_stats, build_raycast_scene, median_edge_length,
                        unsigned_distance)


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
    shifted = unit_box()
    shifted.apply_translation([5.0, 0.0, 0.0])
    scene_shifted = build_raycast_scene(shifted.vertices, shifted.faces)
    ok = alignment_stats(m, scene_aligned, n_samples=100)
    bad = alignment_stats(m, scene_shifted, n_samples=100)
    assert ok["median"] < 1e-4
    assert bad["median"] > 2.0
