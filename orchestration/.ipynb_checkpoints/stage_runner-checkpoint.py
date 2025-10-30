#!/usr/bin/env python3
import argparse, os, json, pandas as pd, subprocess, sys, glob

def run_pre(config, row):
    import glob, subprocess, sys
    lake = row.lake_id
    candidates = glob.glob(f"generated_preprocessing_scripts_*/preprocess_config_lake_{lake}.json")
    if not candidates:
        print(f"[PRE] No per-lake config found for lake {lake}. Provide one or extend this runner.", file=sys.stderr)
        sys.exit(2)
    pre_conf = candidates[0]
    cmd = ["dineof_preprocessor", "--config", pre_conf]  # ← only this line differs
    print("[PRE] Exec:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def run_dineof(config, row):
    lake = row.lake_id
    alpha = row.alpha
    cmd = ["bash", "run_dineof.sh", str(lake), str(alpha)]
    print("[DINEOF] Exec:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def run_post(config, row):
    lake = row.lake_id
    version = row.version
    cmd = ["bash", "run_post.sh", str(lake), version]
    print("[POST] Exec:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pre","dineof","post"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--row", type=int, required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        conf = json.load(f)
    df = pd.read_csv(args.index)
    if not (0 <= args.row < len(df)):
        print(f"Row {args.row} out of bounds 0..{len(df)-1}", file=sys.stderr)
        sys.exit(1)
    row = df.iloc[args.row]

    os.environ.update({
        "LSWT_EXPERIMENT_JSON": os.path.abspath(args.config),
        "LSWT_RUN_INDEX": os.path.abspath(args.index),
        "LSWT_ARRAY_ROW": str(args.row),
        "LSWT_LAKE_ID": str(row.lake_id),
        "LSWT_VERSION": str(row.version),
        "LSWT_ALPHA": str(row.alpha),
    })

    if args.stage == "pre":
        run_pre(conf, row)
    elif args.stage == "dineof":
        run_dineof(conf, row)
    else:
        run_post(conf, row)

if __name__ == "__main__":
    main()