#!/usr/bin/env python3
import argparse, json, os, sys, subprocess, pandas as pd

def main():
    p = argparse.ArgumentParser(description="Submit 3 job arrays (pre→dineof→post) from central JSON.")
    p.add_argument("config", help="Path to experiment_settings.json")
    p.add_argument("--index", default="/mnt/data/run_index.csv", help="Path to run_index.csv (created beforehand)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.config) as f:
        conf = json.load(f)

    submission = conf.get("submission", {})
    max_conc = str(submission.get("max_concurrent", 50))
    log_dir = submission.get("log_dir", "logs")
    partition = submission.get("partition")
    qos = submission.get("qos", conf.get("system_settings", {}).get("qos_choice"))
    time_lim = submission.get("time", "12:00:00")
    mem = submission.get("mem", "8G")
    account = submission.get("account")
    email = submission.get("email")

    os.makedirs(log_dir, exist_ok=True)

    df = pd.read_csv(args.index)
    n_tasks = len(df) - 1
    if n_tasks < 0:
        print("No tasks in index.", file=sys.stderr)
        sys.exit(1)

    base_export = f"CONF={args.config}; INDEX={args.index}"
    def sbatch(script):
        parts = ["sbatch", "--parsable", f"--array=0-{n_tasks}%{max_conc}",
                 "-o", f"{log_dir}/%x_%A_%a.out",
                 "-e", f"{log_dir}/%x_%A_%a.err",
                 f"--time={time_lim}", f"--mem={mem}"]
        if partition: parts += ["-p", partition]
        if qos: parts += ["--qos={qos}"]
        if account: parts += ["-A", account]
        if email: parts += ["--mail-type=FAIL,END", "--mail-user={email}"]
        parts += [script]
        return parts

    pre_cmd  = sbatch("/mnt/data/pre.slurm")
    dino_cmd = sbatch("/mnt/data/dineof.slurm")
    post_cmd = sbatch("/mnt/data/post.slurm")

    if args.dry_run:
        print("DRY RUN — commands I'd execute:")
        print("export", base_export)
        print(" ".join(pre_cmd))
        print(" ".join(dino_cmd), " --dependency=afterok:<PRE_JOB_ID>")
        print(" ".join(post_cmd), " --dependency=afterok:<DINEOF_JOB_ID>")
        return

    env = os.environ.copy()
    env.update({"CONF": args.config, "INDEX": args.index})

    pre_job = subprocess.check_output(pre_cmd, env=env).decode().strip()
    dino_cmd_dep = dino_cmd + [f"--dependency=afterok:{pre_job}"]
    dino_job = subprocess.check_output(dino_cmd_dep, env=env).decode().strip()
    post_cmd_dep = post_cmd + [f"--dependency=afterok:{dino_job}"]
    post_job = subprocess.check_output(post_cmd_dep, env=env).decode().strip()

    print(f"Submitted: pre={pre_job} → dineof={dino_job} → post={post_job}")

if __name__ == "__main__":
    main()