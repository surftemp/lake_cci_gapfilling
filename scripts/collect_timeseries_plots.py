#!/usr/bin/env python3
"""
Collect timeseries and in-situ validation plots into HTML galleries.

Generates galleries for:
  1. DINEOF timeseries
  2. DINEOF vs EOF-filtered comparison
  3. DINEOF vs DINCAE comparison
  4. DINEOF-filtered vs DINCAE comparison
  5. In-situ validation (main multi-method panels per site)
  6. In-situ yearly validation
  7. In-situ same-date comparison

Usage:
  python collect_timeseries_plots.py \
    --exp-dir /gws/.../anomaly-20260131-219f0d-exp0_baseline_both

  # Specific galleries only
  python collect_timeseries_plots.py \
    --exp-dir /gws/.../ --galleries dineof_vs_dincae insitu

  # List available galleries
  python collect_timeseries_plots.py --exp-dir /gws/.../ --list
"""
import argparse, glob, os, base64


# =========================================================================
# Gallery definitions
# =========================================================================

GALLERY_DEFS = {
    # --- Timeseries galleries (in post/*/a*/plots/) ---
    "dineof": {
        "glob": "post/*/a*/plots/LAKE*_DINEOF.png",
        "html": "gallery_dineof.html",
        "title": "DINEOF Timeseries",
        "folder": "timeseries_all_DINEOF",
    },
    "dineof_vs_filtered": {
        "glob": "post/*/a*/plots/LAKE*_DINEOF_vs_DINEOF_filtered.png",
        "html": "gallery_dineof_vs_filtered.html",
        "title": "DINEOF vs EOF-Filtered",
        "folder": "timeseries_all_DINEOF_vs_filtered",
    },
    "dineof_vs_dincae": {
        "glob": "post/*/a*/plots/LAKE*_DINEOF_vs_DINCAE.png",
        "html": "gallery_dineof_vs_dincae.html",
        "title": "DINEOF vs DINCAE",
        "folder": "timeseries_all_DINEOF_vs_DINCAE",
    },
    "filtered_vs_dincae": {
        "glob": "post/*/a*/plots/LAKE*_DINEOF_filtered_vs_DINCAE.png",
        "html": "gallery_filtered_vs_dincae.html",
        "title": "DINEOF-Filtered vs DINCAE",
        "folder": "timeseries_all_filtered_vs_DINCAE",
    },
    # --- In-situ validation galleries (in post/*/a*/insitu_cv_validation/) ---
    "insitu": {
        "glob": "post/*/a*/insitu_cv_validation/LAKE*_insitu_validation_site*.png",
        "html": "gallery_insitu_validation.html",
        "title": "In-Situ Validation (All Methods, Per Site)",
        "folder": "insitu_all_validation",
    },
    "insitu_yearly": {
        "glob": "post/*/a*/insitu_cv_validation/LAKE*_insitu_validation_yearly_*_site*.png",
        "html": "gallery_insitu_yearly.html",
        "title": "In-Situ Yearly Validation",
        "folder": "insitu_all_yearly",
    },
    "insitu_same_date": {
        "glob": "post/*/a*/insitu_cv_validation/LAKE*_same_date_comparison_*_site*.png",
        "html": "gallery_insitu_same_date.html",
        "title": "In-Situ Same-Date Comparison",
        "folder": "insitu_all_same_date",
    },
    "insitu_distance": {
        "glob": "post/*/a*/insitu_cv_validation/LAKE*_distance_map_site*.png",
        "html": "gallery_insitu_distance_map.html",
        "title": "In-Situ Buoy Distance Maps",
        "folder": "insitu_all_distance_maps",
    },
}


# =========================================================================
# HTML builder
# =========================================================================

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
                "<h1>%s (%d plots)</h1>\n" % (title, len(plots)))

        for p_path in plots:
            label = os.path.basename(p_path).replace(".png", "")
            if use_resize and resize_fn is not None:
                b64, fmt = resize_fn(p_path)
            else:
                with open(p_path, "rb") as img:
                    b64 = base64.b64encode(img.read()).decode()
                fmt = "png"
            f.write(f'<div class="card"><h2>{label}</h2>\n')
            f.write(f'<img src="data:image/{fmt};base64,{b64}"/></div>\n')

        f.write("</body></html>")
    size_mb = os.path.getsize(html_path) / 1e6
    print(f"  Wrote {html_path} ({size_mb:.1f} MB)")


# =========================================================================
# Main
# =========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Collect pipeline plots into HTML galleries for visual inspection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--exp-dir", required=True, help="Experiment root directory")
    p.add_argument("--galleries", nargs="+", choices=list(GALLERY_DEFS.keys()),
                   help="Build only these galleries (default: all)")
    p.add_argument("--list", action="store_true", help="List available galleries and exit")
    p.add_argument("--no-resize", action="store_true",
                   help="Skip PIL resize, embed full-size PNGs")
    args = p.parse_args()

    if args.list:
        print("Available galleries:")
        for key, defn in GALLERY_DEFS.items():
            print(f"  {key:25s} {defn['title']}")
        return

    exp = args.exp_dir.rstrip("/")

    # Setup resize
    resize_fn = None
    use_resize = False
    if not args.no_resize:
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
            print("PIL available -- resizing to max 900px width, JPEG q70")
        except ImportError:
            print("PIL not available -- embedding full-size images")

    # Select galleries
    selected = args.galleries or list(GALLERY_DEFS.keys())
    total_built = 0

    for key in selected:
        defn = GALLERY_DEFS[key]
        plots = sorted(glob.glob(os.path.join(exp, defn["glob"])))
        print(f"\n{defn['title']}: found {len(plots)} plots")
        if not plots:
            continue

        out_dir = os.path.join(exp, defn["folder"])
        os.makedirs(out_dir, exist_ok=True)

        # Symlink into flat dir
        for p_path in plots:
            dest = os.path.join(out_dir, os.path.basename(p_path))
            if os.path.islink(dest) or os.path.exists(dest):
                os.remove(dest)
            os.symlink(p_path, dest)
        print(f"  Symlinked {len(plots)} files to {out_dir}/")

        build_gallery(plots, out_dir, defn["html"], defn["title"], use_resize, resize_fn)
        total_built += 1

    print(f"\nDone. Built {total_built} galleries.")


if __name__ == "__main__":
    main()
