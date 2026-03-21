#!/usr/bin/env python3
"""
Collect timeseries and in-situ validation plots into HTML galleries.

Generates galleries for:
  1. DINEOF timeseries
  2. DINEOF vs EOF-filtered comparison
  3. DINEOF vs DINCAE comparison
  4. DINEOF-filtered vs DINCAE comparison
  5. DINEOF-filtered-interp vs DINCAE-interp (side-by-side)
  6. DINEOF-interp vs DINEOF-filtered-interp (side-by-side)
  7. In-situ validation (main multi-method panels per site)
  8. In-situ yearly validation
  9. In-situ same-date comparison
  10. In-situ distance maps

Usage:
  python collect_timeseries_plots.py \
    --exp-dir /gws/.../anomaly-20260131-219f0d-exp0_baseline_both

  # Specific galleries only
  python collect_timeseries_plots.py \
    --exp-dir /gws/.../ --galleries filtered_interp_vs_dincae_interp

  # List available galleries
  python collect_timeseries_plots.py --exp-dir /gws/.../ --list
"""
import argparse, glob, os, base64, re


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
    # --- Interpolated comparison galleries (side-by-side from individual PNGs) ---
    "filtered_interp_vs_dincae_interp": {
        "type": "paired",
        "glob_left": "post/*/a*/plots/LAKE*_DINEOF_filtered_interp.png",
        "glob_right": "post/*/a*/plots/LAKE*_DINCAE_interp.png",
        "label_left": "DINEOF Filtered Interp",
        "label_right": "DINCAE Interp",
        "html": "gallery_filtered_interp_vs_dincae_interp.html",
        "title": "DINEOF-Filtered-Interp vs DINCAE-Interp",
        "folder": "timeseries_all_filtered_interp_vs_DINCAE_interp",
    },
    "dineof_interp_vs_filtered_interp": {
        "type": "paired",
        "glob_left": "post/*/a*/plots/LAKE*_DINEOF_interp.png",
        "glob_right": "post/*/a*/plots/LAKE*_DINEOF_filtered_interp.png",
        "label_left": "DINEOF Interp",
        "label_right": "DINEOF Filtered Interp",
        "html": "gallery_dineof_interp_vs_filtered_interp.html",
        "title": "DINEOF-Interp vs DINEOF-Filtered-Interp",
        "folder": "timeseries_all_DINEOF_interp_vs_filtered_interp",
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


def _extract_lake_key(path):
    """Extract lake ID + alpha from path for pairing."""
    # e.g. post/000000002/a0.01/plots/LAKE000000002_DINEOF_interp.png
    # key = "000000002/a0.01"
    m = re.search(r'post/(\d+)/(a[\d.]+)/plots/', path)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return os.path.basename(path)


def _embed_image(path):
    """Read image and return base64-encoded PNG (no resize, full quality)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return b64


# =========================================================================
# HTML builders
# =========================================================================

def _gallery_style():
    return ("<!DOCTYPE html><html><head><style>\n"
            "body{font-family:sans-serif;background:#111;color:#eee;margin:20px}\n"
            "h1{text-align:center}\n"
            ".card{margin:30px auto;max-width:95%%;background:#222;padding:15px;border-radius:8px}\n"
            ".card h2{margin:5px 0;font-size:16px}\n"
            ".card img{width:100%%;height:auto}\n"
            ".pair{display:flex;gap:10px;align-items:flex-start}\n"
            ".pair .side{flex:1;min-width:0}\n"
            ".pair .side h3{margin:5px 0;font-size:14px;text-align:center}\n"
            ".pair .side img{width:100%%;height:auto}\n"
            "</style></head><body>\n")


def build_gallery(plots, out_dir, html_name, title):
    """Build a single HTML gallery from a list of plot paths. Full resolution PNG."""
    html_path = os.path.join(out_dir, html_name)
    with open(html_path, "w") as f:
        f.write(_gallery_style())
        f.write(f"<h1>{title} ({len(plots)} plots)</h1>\n")

        for p_path in plots:
            label = os.path.basename(p_path).replace(".png", "")
            b64 = _embed_image(p_path)
            f.write(f'<div class="card"><h2>{label}</h2>\n')
            f.write(f'<img src="data:image/png;base64,{b64}"/></div>\n')

        f.write("</body></html>")
    size_mb = os.path.getsize(html_path) / 1e6
    print(f"  Wrote {html_path} ({size_mb:.1f} MB)")


def build_paired_gallery(pairs, out_dir, html_name, title, label_left, label_right):
    """Build side-by-side HTML gallery from paired plot paths. Full resolution PNG."""
    html_path = os.path.join(out_dir, html_name)
    with open(html_path, "w") as f:
        f.write(_gallery_style())
        f.write(f"<h1>{title} ({len(pairs)} pairs)</h1>\n")

        for key, left_path, right_path in pairs:
            lake_label = re.search(r'LAKE(\d+)', os.path.basename(left_path))
            lake_str = f"Lake {lake_label.group(1)}" if lake_label else key
            f.write(f'<div class="card"><h2>{lake_str} ({key})</h2>\n')
            f.write('<div class="pair">\n')

            # Left
            b64_l = _embed_image(left_path)
            f.write(f'<div class="side"><h3>{label_left}</h3>\n')
            f.write(f'<img src="data:image/png;base64,{b64_l}"/></div>\n')

            # Right
            b64_r = _embed_image(right_path)
            f.write(f'<div class="side"><h3>{label_right}</h3>\n')
            f.write(f'<img src="data:image/png;base64,{b64_r}"/></div>\n')

            f.write('</div></div>\n')

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
    args = p.parse_args()

    if args.list:
        print("Available galleries:")
        for key, defn in GALLERY_DEFS.items():
            print(f"  {key:45s} {defn['title']}")
        return

    exp = args.exp_dir.rstrip("/")
    selected = args.galleries or list(GALLERY_DEFS.keys())
    total_built = 0

    for key in selected:
        defn = GALLERY_DEFS[key]
        is_paired = defn.get("type") == "paired"

        if is_paired:
            # Paired gallery: match left/right by lake+alpha key
            left_plots = sorted(glob.glob(os.path.join(exp, defn["glob_left"])))
            right_plots = sorted(glob.glob(os.path.join(exp, defn["glob_right"])))
            print(f"\n{defn['title']}: found {len(left_plots)} left, {len(right_plots)} right")

            # Index right plots by key
            right_by_key = {}
            for rp in right_plots:
                right_by_key[_extract_lake_key(rp)] = rp

            pairs = []
            for lp in left_plots:
                lk = _extract_lake_key(lp)
                if lk in right_by_key:
                    pairs.append((lk, lp, right_by_key[lk]))

            print(f"  Matched {len(pairs)} pairs")
            if not pairs:
                continue

            out_dir = os.path.join(exp, defn["folder"])
            os.makedirs(out_dir, exist_ok=True)

            build_paired_gallery(
                pairs, out_dir, defn["html"], defn["title"],
                defn["label_left"], defn["label_right"],
            )
            total_built += 1

        else:
            # Standard single-image gallery
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

            build_gallery(plots, out_dir, defn["html"], defn["title"])
            total_built += 1

    print(f"\nDone. Built {total_built} galleries.")


if __name__ == "__main__":
    main()
