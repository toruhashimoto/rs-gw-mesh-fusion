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
