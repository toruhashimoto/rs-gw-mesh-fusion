"""Render fusion previews with nvdiffrast (CUDA context).

RS geometry is painted gray, GW complement patches orange, so the added
regions can be inspected before importing the fused mesh into RealityScan.
Requires the MSVC/CUDA runtime env (run via run_fuse-style launcher or after
the nvdiffrast JIT cache is warm).
"""
import argparse
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
    ap.add_argument("--rs", required=True)
    ap.add_argument("--gw", required=True)
    ap.add_argument("--patch_npy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    import nvdiffrast.torch as dr

    rs = trimesh.load(args.rs, process=False)
    gw = trimesh.load(args.gw, process=False)
    patch_idx = np.load(args.patch_npy)

    rs_v = np.asarray(rs.vertices, dtype=np.float32)
    rs_f = np.asarray(rs.faces, dtype=np.int64)
    patch = gw.submesh([patch_idx], append=True) if len(patch_idx) else None

    verts = [rs_v]
    faces = [rs_f]
    cols = [np.tile(RS_GRAY, (len(rs_v), 1))]
    if patch is not None:
        pv = np.asarray(patch.vertices, dtype=np.float32)
        faces.append(np.asarray(patch.faces, dtype=np.int64) + len(rs_v))
        verts.append(pv)
        cols.append(np.tile(PATCH_ORANGE, (len(pv), 1)))

    v = torch.tensor(np.concatenate(verts), dtype=torch.float32, device="cuda")
    f = torch.tensor(np.concatenate(faces), dtype=torch.int32, device="cuda")
    c = torch.tensor(np.concatenate(cols), dtype=torch.float32, device="cuda")

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
