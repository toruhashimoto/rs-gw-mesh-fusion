"""MeshFusion desktop app (local Gradio UI).

Wraps fuse_meshes.py: pick the RealityScan High Detail mesh and the
complement mesh (e.g. GaussianWrapping output), tune parameters, run, and
inspect the report / previews. The fusion core is CPU-only; previews need an
optional CUDA + nvdiffrast setup and are skipped gracefully without it.
"""
import os
import subprocess
import sys

import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
FUSE = os.path.join(HERE, "fuse_meshes.py")
RENDER = os.path.join(HERE, "render_compare.py")


def _stream(cmd):
    """Run a command, yielding accumulated stdout as it arrives."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        yield "\n".join(lines), proc
    proc.wait()
    yield "\n".join(lines), proc


def run_fusion(rs_path, gw_path, out_dir, roi_json, tau_factor, tau_abs,
               roi_expand, min_patch_area_ratio, overlap_rings,
               use_icp, do_clean, export_obj, make_preview):
    log = ""
    report = ""
    img1 = img2 = None

    def state(msg):
        return (log + ("\n" if log else "") + msg, report, img1, img2)

    for p, name in [(rs_path, "RS mesh"), (gw_path, "complement mesh")]:
        if not p or not os.path.isfile(p):
            yield state(f"[ERROR] {name} not found: {p!r}")
            return
    if not out_dir:
        yield state("[ERROR] output directory is empty")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Gradio delivers None for cleared/untouched number fields - fall back to
    # the CLI defaults instead of emitting "None" into the command line.
    cmd = [sys.executable, FUSE, "--rs", rs_path, "--gw", gw_path, "--out", out_dir,
           "--tau_factor", str(tau_factor if tau_factor is not None else 8.0),
           "--roi_expand", str(roi_expand if roi_expand is not None else 0.10),
           "--min_patch_area_ratio",
           str(min_patch_area_ratio if min_patch_area_ratio is not None else 1e-4),
           "--overlap_rings", str(int(overlap_rings if overlap_rings is not None else 3))]
    if tau_abs and float(tau_abs) > 0:
        cmd += ["--tau", str(tau_abs)]
    if roi_json:
        if not os.path.isfile(roi_json):
            yield state(f"[ERROR] ROI JSON not found: {roi_json!r}")
            return
        cmd += ["--roi_json", roi_json]
    if use_icp:
        cmd.append("--icp")
    if do_clean:
        cmd.append("--clean")
    if export_obj:
        cmd.append("--obj")

    proc = None
    for log, proc in _stream(cmd):
        yield (log, report, img1, img2)
    if proc.returncode != 0:
        yield state(f"[ERROR] fusion exited with code {proc.returncode} - see log above")
        return

    report_path = os.path.join(out_dir, "fusion_report.txt")
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = f.read()

    if make_preview:
        yield state("[INFO] rendering previews (needs torch + nvdiffrast; skipped on failure)...")
        pv = subprocess.run(
            [sys.executable, RENDER,
             "--fused", os.path.join(out_dir, "fused.ply"),
             "--meta", os.path.join(out_dir, "fusion_meta.json"),
             "--out", out_dir],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if pv.returncode == 0:
            p1 = os.path.join(out_dir, "preview_view1.png")
            p2 = os.path.join(out_dir, "preview_view2.png")
            img1 = p1 if os.path.isfile(p1) else None
            img2 = p2 if os.path.isfile(p2) else None
            log += "\n[INFO] previews rendered"
        else:
            tail = (pv.stdout or "").strip().splitlines()[-3:]
            log += ("\n[WARN] preview rendering unavailable (this is optional): "
                    + " / ".join(tail))
    yield (log + "\n[DONE] fused.ply is ready for RealityScan import", report, img1, img2)


with gr.Blocks(title="MeshFusion") as demo:
    gr.Markdown("# MeshFusion\n"
                "Fuse a **RealityScan High Detail mesh** (primary, kept untouched) with a "
                "**complement mesh** (e.g. Gaussian Wrapping output) into one model, "
                "then re-import it into RealityScan and run Texture on it.")
    with gr.Row():
        with gr.Column():
            rs_path = gr.Textbox(label="RealityScan High Detail mesh (.ply / .obj)",
                                 placeholder=r"C:\path\to\rs_high_detail.ply")
            gw_path = gr.Textbox(label="Complement mesh (.ply)",
                                 placeholder=r"C:\path\to\gaussian_wrapping_mesh.ply")
            out_dir = gr.Textbox(label="Output directory",
                                 placeholder=r"C:\path\to\output")
            roi_json = gr.Textbox(label="ROI bounding volume JSON (optional, "
                                        "Gaussian Wrapping Blender add-on format)")
            with gr.Accordion("Parameters", open=False):
                tau_factor = gr.Slider(1, 20, value=8, step=0.5,
                                       label="tau factor (x RS median edge length)")
                tau_abs = gr.Number(value=0, label="tau absolute override (0 = auto)")
                roi_expand = gr.Slider(0.0, 0.5, value=0.10, step=0.01,
                                       label="ROI expansion (x RS AABB span)")
                min_patch_area_ratio = gr.Number(value=1e-4,
                                                 label="min patch area ratio (x RS area)")
                overlap_rings = gr.Slider(0, 10, value=3, step=1, label="overlap rings")
            use_icp = gr.Checkbox(value=True, label="ICP alignment (recommended: RealityScan "
                                                    "mesh/COLMAP exports use different frames)")
            do_clean = gr.Checkbox(value=False, label="clean degenerate/duplicate faces")
            export_obj = gr.Checkbox(value=False, label="also export fused.obj (large)")
            make_preview = gr.Checkbox(value=True, label="render previews (needs CUDA GPU)")
            run_btn = gr.Button("Fuse", variant="primary")
        with gr.Column():
            log_box = gr.Textbox(label="Log", lines=18, max_lines=18, autoscroll=True)
            report_box = gr.Textbox(label="Fusion report", lines=8, max_lines=12)
    with gr.Row():
        img1 = gr.Image(label="Preview 1 (gray=RS, orange=complement)", type="filepath")
        img2 = gr.Image(label="Preview 2", type="filepath")

    run_btn.click(run_fusion,
                  inputs=[rs_path, gw_path, out_dir, roi_json, tau_factor, tau_abs,
                          roi_expand, min_patch_area_ratio, overlap_rings,
                          use_icp, do_clean, export_obj, make_preview],
                  outputs=[log_box, report_box, img1, img2])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", inbrowser=True)
