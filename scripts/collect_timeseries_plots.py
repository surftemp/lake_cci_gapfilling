#!/usr/bin/env python3
"""
Collect timeseries plots into HTML galleries for visual inspection.

Generates two galleries:
  1. DINEOF timeseries (LAKE*_DINEOF.png)
  2. DINEOF vs EOF-filtered comparison (LAKE*_DINEOF_vs_DINEOF_filtered.png)

Usage:
  python collect_timeseries_plots.py \
    --exp-dir /gws/.../anomaly-20260131-219f0d-exp0_baseline_both
"""
import argparse, glob, os, base64


def build_gallery(plots, out_dir, html_name, title, use_resize, resize_fn):
    """Build a single HTML gallery from a list of plot paths."""
    html_path = os.path.join(out_dir, html_name)
    with open(html_path, "w") as f:
        f.write("<!DOCTYPE html><html><head><style>\n"
               "body{font-family:sans-serif;background:#111;color:#eee;margin:20px}\n"
               "h1{text-align:center}\n"
               ".card{margin:30px auto;max-width:1400px;background:#222;padding:15px;border-radius:8px}\n"
               ".card h2{margin:5px 0;font-size:16px}\n"
               ".card img{width:100%%;height:auto}\n"
               "</style></head><body>\n"
               "<h1>%s (%d lakes)</h1>\n" % (title, len(plots)))

        for p_path in plots:
            lake = os.path.basename(p_path).replace(".png", "")
            if use_resize and resize_fn is not None:
                b64, fmt = resize_fn(p_path)
            else:
                with open(p_path, "rb") as img:
                    b64 = base64.b64encode(img.read()).decode()
                fmt = "png"
            f.write(f'<div class="card"><h2>{lake}</h2>\n')
            f.write(f'<img src="data:image/{fmt};base64,{b64}"/></div>\n')

        f.write("</body></html>")
    size_mb = os.path.getsize(html_path) / 1e6
    print(f"  Wrote {html_path} ({size_mb:.1f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", required=True)
    args = p.parse_args()

    exp = args.exp_dir.rstrip("/")

    # Setup resize
    resize_fn = None
    use_resize = False
    try:
        from PIL import Image
        import io
        def resize_to_b64(path, max_width=900):
            img = Image.open(path).convert("RGB")
            if img.width > max_width:
                ratio = max_width / img.width
                img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=70)
            return base64.b64encode(buf.getvalue()).decode(), "jpeg"
        resize_fn = resize_to_b64
        use_resize = True
        print("PIL available — resizing to max 900px width, JPEG q70")
    except ImportError:
        print("PIL not available — embedding full-size images")

    # Gallery definitions: (glob_pattern, html_filename, title, output_folder)
    galleries = [
        ("LAKE*_DINEOF.png",
         "gallery_dineof.html",
         "DINEOF Timeseries Gallery",
         "timeseries_all_DINEOF"),
        ("LAKE*_DINEOF_vs_DINEOF_filtered.png",
         "gallery_dineof_vs_filtered.html",
         "DINEOF vs EOF-Filtered Gallery",
         "timeseries_all_DINEOF_vs_filtered"),
    ]

    for pattern, html_name, title, out_folder in galleries:
        plots = sorted(glob.glob(f"{exp}/post/*/a*/plots/{pattern}"))
        print(f"\n{title}: found {len(plots)} plots")
        if not plots:
            continue

        out_dir = os.path.join(exp, out_folder)
        os.makedirs(out_dir, exist_ok=True)

        # Symlink into flat dir
        for p_path in plots:
            dest = os.path.join(out_dir, os.path.basename(p_path))
            if os.path.islink(dest) or os.path.exists(dest):
                os.remove(dest)
            os.symlink(p_path, dest)
        print(f"  Symlinked to {out_dir}/")

        build_gallery(plots, out_dir, html_name, title, use_resize, resize_fn)

    print("\nDone.")


if __name__ == "__main__":
    main()
