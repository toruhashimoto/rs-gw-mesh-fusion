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


def icp_refine(rs_mesh, gw_mesh, initial_median, med_edge, n_sample=300_000, seed=0):
    """Multi-scale scaled ICP registering the GW mesh onto the RS mesh.

    The correspondence-distance schedule starts above the measured
    misalignment and shrinks toward the RS edge length; a single-shot ICP
    with a small radius cannot converge when the initial offset is large
    (validated on Sample_RS-ply: median 5.6 -> 0.009).
    Source points are GW vertices inside the (expanded) RS AABB so that GW
    background geometry does not pull the registration.
    """
    import open3d as o3d
    rng = np.random.default_rng(seed)

    def sample_pcd(points):
        idx = rng.choice(len(points), min(n_sample, len(points)), replace=False)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[idx])
        return pcd

    gv = gw_mesh.vertices.view(np.ndarray)
    span = rs_mesh.bounds[1] - rs_mesh.bounds[0]
    inb = np.all((gv >= rs_mesh.bounds[0] - 0.1 * span) &
                 (gv <= rs_mesh.bounds[1] + 0.1 * span), axis=1)
    src_pts = gv[inb] if inb.sum() > 10_000 else gv
    src = sample_pcd(src_pts)
    dst = sample_pcd(rs_mesh.vertices.view(np.ndarray))

    mc0 = max(1.5 * initial_median, 8.0 * med_edge)
    schedule = [max(mc0 / (4 ** k), 4.0 * med_edge) for k in range(4)]
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True)
    T = np.eye(4)
    fitness = rmse = 0.0
    for mc in schedule:
        res = o3d.pipelines.registration.registration_icp(
            src, dst, mc, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
        T, fitness, rmse = res.transformation, float(res.fitness), float(res.inlier_rmse)
    return np.asarray(T), fitness, rmse


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    os.makedirs(args.out, exist_ok=True)
    lines = []

    def log(msg):
        print(msg, flush=True)
        lines.append(msg)

    def write_report():
        with open(os.path.join(args.out, "fusion_report.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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
    T_applied = None
    if st["median"] > align_thresh and args.icp:
        T, fitness, rmse = icp_refine(rs, gw, st["median"], rs_med_edge, seed=args.seed)
        gw.apply_transform(T)
        T_applied = T
        scale = float(np.cbrt(np.linalg.det(T[:3, :3])))
        log(f"[INFO] multi-scale ICP applied (fitness={fitness:.3f} rmse={rmse:.6g} "
            f"scale={scale:.6f}); rechecking")
        gw_scene = fl.build_raycast_scene(gw.vertices, gw.faces)
        st = fl.alignment_stats(rs, gw_scene, seed=args.seed)
        log(f"[INFO] alignment after ICP: median={st['median']:.6g}")
    if st["median"] > align_thresh:
        log("[ERROR] Meshes do not appear to share a coordinate frame. "
            "Verify the RS export settings, or re-run with --icp.")
        write_report()
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
    else:
        log(f"[INFO] patches: {n_patch:,} faces, area {patch_area:.6g} "
            f"({patch_area / rs.area:.2%} of RS area)")
    ply_path = os.path.join(args.out, "fused.ply")
    fused.export(ply_path)
    log(f"[INFO] wrote {ply_path} (V={len(fused.vertices):,} F={len(fused.faces):,})")
    np.save(os.path.join(args.out, "patch_faces.npy"), np.flatnonzero(mask))
    meta = {"n_rs_faces": int(len(rs.faces)), "n_patch_faces": int(n_patch),
            "gw_to_rs_transform": (np.asarray(T_applied).tolist()
                                   if T_applied is not None else None)}
    with open(os.path.join(args.out, "fusion_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if args.obj:
        obj_path = os.path.join(args.out, "fused.obj")
        fused.export(obj_path)
        log(f"[INFO] wrote {obj_path}")

    write_report()
    return {"n_patch_faces": n_patch, "patch_area": patch_area,
            "alignment": st, "tau": tau, "components_kept": kept,
            "components_removed": removed}


if __name__ == "__main__":
    main()
