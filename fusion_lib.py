"""Core mesh-fusion primitives: BVH distances, edge stats, alignment check,
patch selection (distance / ROI / connected components / dilation), and fusion.

All face masks are boolean arrays over the GW mesh's faces. The RS mesh is
never modified.
"""
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
