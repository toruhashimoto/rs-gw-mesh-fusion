"""Render fusion previews with nvdiffrast (CUDA context).

Reads the fused mesh + fusion_meta.json: the first n_rs_faces faces (RS part)
are painted gray, the remaining patch faces orange, so the added regions can
be inspected before importing the fused mesh into RealityScan.
Requires the MSVC/CUDA runtime env (run via a run_fuse-style launcher or after
the nvdiffrast JIT cache is warm).
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import trimesh

RS_GRAY = np.array([180, 180, 180], dtype=np.float32) / 255.0
PATCH_ORANGE = np.array([255, 140, 0], dtype=np.float32) / 255.0


def look_at(eye, at, up):
    f = (at - eye)
    f = f / f.norm()
    s = torch.linalg.cross(f, up)
    s = s / s.norm()
    u = torch.linalg.cross(s, f)
    mv = torch.eye(4, device="cuda")
    mv[0, :3], mv[1, :3], mv[2, :3] = s, u, -f
    mv[:3, 3] = -mv[:3, :3] @ eye
    return mv


def perspective(fovy_deg, aspect, near, far):
    t = 1.0 / np.tan(np.radians(fovy_deg) / 2)
    p = torch.zeros(4, 4, device="cuda")
    p[0, 0], p[1, 1] = t / aspect, t
    p[2, 2] = (far + near) / (near - far)
    p[2, 3] = 2 * far * near / (near - far)
    p[3, 2] = -1.0
    return p


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", required=True, help="fused.ply from fuse_meshes.py")
    ap.add_argument("--meta", required=True, help="fusion_meta.json from fuse_meshes.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    import nvdiffrast.torch as dr

    fused = trimesh.load(args.fused, process=False)
    with open(args.meta, encoding="utf-8") as fmeta:
        meta = json.load(fmeta)
    n_rs = int(meta["n_rs_faces"])

    faces_all = np.asarray(fused.faces, dtype=np.int64)
    rs_part = trimesh.Trimesh(fused.vertices, faces_all[:n_rs], process=False)
    patch_part = (trimesh.Trimesh(fused.vertices, faces_all[n_rs:], process=False)
                  if len(faces_all) > n_rs else None)

    # nvdiffrast's CudaRaster supports ~2^24 triangles per draw; decimate the
    # RS block (preview only) to stay under the limit, keep patches full-res.
    MAX_TRIS = 14_000_000
    n_patch = 0 if patch_part is None else len(patch_part.faces)
    if n_rs + n_patch > MAX_TRIS:
        target = MAX_TRIS - n_patch
        print(f"[INFO] decimating RS block for preview: {n_rs:,} -> {target:,} faces")
        rs_part = rs_part.simplify_quadric_decimation(face_count=target)

    verts = [np.asarray(rs_part.vertices, dtype=np.float32)]
    faces = [np.asarray(rs_part.faces, dtype=np.int64)]
    cols = [np.tile(RS_GRAY, (len(rs_part.vertices), 1)).astype(np.float32)]
    if patch_part is not None:
        pv = np.asarray(patch_part.vertices, dtype=np.float32)
        faces.append(np.asarray(patch_part.faces, dtype=np.int64) + len(verts[0]))
        verts.append(pv)
        cols.append(np.tile(PATCH_ORANGE, (len(pv), 1)).astype(np.float32))
    v_np = np.concatenate(verts)
    f_np = np.concatenate(faces)
    colors = np.concatenate(cols)

    v = torch.tensor(v_np, dtype=torch.float32, device="cuda")
    f = torch.tensor(f_np, dtype=torch.int32, device="cuda")
    c = torch.tensor(colors, dtype=torch.float32, device="cuda")

    center = v.mean(dim=0)
    ext = float((v.max(dim=0).values - v.min(dim=0).values).norm())
    ctx = dr.RasterizeCudaContext()
    proj = perspective(55.0, args.width / args.height, ext * 0.01, ext * 10)
    up = torch.tensor([0.0, -1.0, 0.0], device="cuda")

    for ang, tag in [(0.35, "view1"), (2.45, "view2")]:
        eye = center + ext * 0.40 * torch.tensor(
            [np.sin(ang), -0.25, np.cos(ang)], dtype=torch.float32, device="cuda")
        mvp = proj @ look_at(eye, center, up)
        v_hom = torch.cat([v, torch.ones_like(v[:, :1])], dim=1)
        v_clip = (mvp @ v_hom.T).T.unsqueeze(0).contiguous()
        rast, _ = dr.rasterize(ctx, v_clip, f, resolution=[args.height, args.width])
        color, _ = dr.interpolate(c.unsqueeze(0).contiguous(), rast, f)
        img = color[0].clamp(0, 1).cpu().numpy()
        alpha = (rast[0, ..., 3:4] > 0).float().cpu().numpy()
        img = img * alpha + (1 - alpha)  # white background
        path = os.path.join(args.out, f"preview_{tag}.png")
        imageio.imwrite(path, (img * 255).astype(np.uint8))
        print(f"[INFO] saved {path} coverage={float(alpha.mean()):.2f}")
    print("[INFO] PREVIEW DONE")


if __name__ == "__main__":
    main()
