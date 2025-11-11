#!/usr/bin/env python3
# Unified LSWT controller with auto run_tag, manifest, and single-JSON control.
# Modes:
#  - submission.per_index_chain = true  -> one array; each index runs selected stages inline
#  - submission.per_index_chain = false -> arrays with stage-wide dependencies
#
# Engines: engine_mode = "dineof" | "dincae" | "both"
#
# Usage:
#   python lswtctl.py plan   configs/experiment_settings.json
#   python lswtctl.py submit configs/experiment_settings.json
#   python lswtctl.py exec --config <json> --row <i> --stage <pre|dineof|dincae|post_dineof|post_dincae|chain>
#   python lswtctl.py paths --config <json> --row <i>

import argparse, json, os, sys, subprocess, tempfile, pathlib, itertools, hashlib, shutil
from datetime import datetime, timezone
from dincae_arm import PreparedNC, build_inputs as dincae_build_inputs
from dincae_arm import run_dincae as dincae_run
from dincae_arm import write_dineof_shaped_outputs as dincae_write_out

# ---------- small utils ----------

def _fmt_ids(lake_id:int):
    return lake_id, f"{lake_id:09d}"

def _render(tpl:str, run_tag:str, lake_id:int):
    lake, lake9 = _fmt_ids(lake_id)
    return tpl.format(lake_id=lake, lake_id9=lake9, run_tag=run_tag)

def _ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

def _ensure_test_id(cfg: dict, run_tag: str) -> None:
    if not cfg.get("test_id"):
        cfg["test_id"] = run_tag

def _bash_exec(cmd: str, stage_env: str | None):
    if stage_env:
        cmd = f"{stage_env} && {cmd}"
    return subprocess.check_call(["/bin/bash", "-l", "-c", cmd])

def _grid(conf):
    ds = conf.get("dataset_options", {})
    lakes  = ds.get("custom_lake_ids", [])
    vers   = ds.get("versions_to_process", ["v2"])
    alphas = ds.get("alpha_values", conf.get("dineof_parameters", {}).get("alpha_values", [0.1]))
    return list(itertools.product(lakes, vers, alphas))

def _canonical_dump(c: dict) -> str:
    remove = {"submission", "env"}
    base = {k: v for k, v in c.items() if k not in remove}
    return json.dumps(base, sort_keys=True, separators=(",", ":"))

def _auto_run_tag(conf: dict) -> str:
    mode = conf.get("mode", "anomaly")
    note = conf.get("note", "")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    h = hashlib.sha1(_canonical_dump(conf).encode()).hexdigest()[:6]
    tag = f"{mode}-{date}-{h}"
    return f"{tag}-{note}" if note else tag

def _alpha_slug(a: float) -> str:
    s = f"{a:.6f}".rstrip("0").rstrip(".")
    frac = s.split(".")[1] if "." in s else "0"
    return "a" + frac.ljust(4, "0")

def _resolve_paths(conf, lake_id:int, alpha:float):
    P   = conf.get("paths", {})
    tag = P.get("run_tag") or _auto_run_tag(conf)
    run_root_tpl = P["run_root_template"]
    run_root = _render(run_root_tpl, tag, lake_id)

    alpha_slug = _alpha_slug(alpha)
    def subfmt(tpl: str) -> str:
        s = tpl.replace("{run_root}", run_root).replace("{alpha_slug}", alpha_slug)
        return _render(s, tag, lake_id)

    prep_dir   = subfmt(P["prepared_dir_template"])
    dineof_dir = subfmt(P["output_dir_template"])  # legacy name = dineof
    dincae_dir = subfmt(P.get("dincae_dir_template", "{run_root}/dincae/{lake_id9}/{alpha_slug}"))
    post_dir   = subfmt(P["post_dir_template"])
    html_dir   = subfmt(P["html_dir_template"])
    logs_dir   = subfmt(P.get("logs_dir_template", "{run_root}/logs"))

    lake_ts  = _render(P["lake_ts_template"], tag, lake_id)
    clim_nc  = _render(P["climatology_template"], tag, lake_id)

    prepared_name = P.get("prepared_filename", "prepared.nc")
    prepared_nc   = os.path.join(prep_dir, prepared_name)

    # engine-separated result files
    results_nc_dineof = os.path.join(dineof_dir, "dineof_results.nc")
    results_nc_dincae = os.path.join(dincae_dir, "dincae_results.nc")  # adaptor will copy/shape to here

    _, lake9 = _fmt_ids(lake_id)
    front = f"LAKE{lake9}-CCI-L3S-LSWT-CDR-4.5-filled_fine"
    post_dineof = os.path.join(post_dir, f"{front}_dineof.nc")
    post_dincae = os.path.join(post_dir, f"{front}_dincae.nc")

    return {
        "run_root": run_root, "run_tag": tag, "logs_dir": logs_dir,
        "prepared_dir": prep_dir, "dineof_dir": dineof_dir, "dincae_dir": dincae_dir,
        "post_dir": post_dir, "html_dir": html_dir,
        "prepared_nc": prepared_nc,
        "results_nc_dineof": results_nc_dineof, "results_nc_dincae": results_nc_dincae,
        "lake_ts": lake_ts, "clim_nc": clim_nc,
        "post_dineof": post_dineof, "post_dincae": post_dincae,
        "alpha_slug": alpha_slug, "front": front
    }

def _idempotent_skip(path:str, label:str):
    if os.path.isfile(path):
        print(f"[{label}] Exists → {path} (skip)")
        return True
    return False

# ---------- stage.slurm materializer ----------

def _ensure_stage_slurm():
    """Create minimal stage.slurm if missing/invalid. Supports extended STAGE set."""
    stage_path = pathlib.Path(__file__).parent / "stage.slurm"
    needs_create = True
    if stage_path.exists():
        try:
            first_line = stage_path.read_text().split('\n')[0]
            if first_line.strip() == '#!/bin/bash':
                needs_create = False
        except:
            pass
    if not needs_create:
        return

    stage_content = r"""#!/bin/bash
set -euo pipefail
module purge || true

LAKE_ID=$(python - <<'PY'
import json,itertools,os,sys
with open(os.environ['CONF']) as f:
    conf=json.load(f)
ds=conf.get('dataset_options',{})
l=ds.get('custom_lake_ids',[])
v=ds.get('versions_to_process',['v2'])
a=ds.get('alpha_values',conf.get('dineof_parameters',{}).get('alpha_values',[0.1]))
grid=list(itertools.product(l,v,a))
i=int(os.environ['SLURM_ARRAY_TASK_ID'])
print(grid[i][0] if i < len(grid) else "UNKNOWN")
PY
)

OUTBASE="${LOGS_DIR}/${STAGE}_lake${LAKE_ID}_row${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}"
OUTFILE="${OUTBASE}.out}"
ERRFILE="${OUTBASE}.err}"
exec 1> "$OUTFILE"
exec 2> "$ERRFILE"

echo "[$(date)] Starting ${STAGE} lake_id=${LAKE_ID} row=${SLURM_ARRAY_TASK_ID}"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "stdout: $OUTFILE"
echo "stderr: $ERRFILE"
echo ""

run_stage () {
  local stage_name="$1"
  echo "[CHAIN] >>> ${stage_name}"
  python lswtctl.py exec --config "$CONF" --row "$SLURM_ARRAY_TASK_ID" --stage "${stage_name}"
  echo "[CHAIN] <<< ${stage_name}"
}

if [[ "${STAGE}" == "chain" ]]; then
  # Decide chain sequence based on engine_mode
  EMODE=$(python - <<'PY'
import json,os
with open(os.environ['CONF']) as f: conf=json.load(f)
print(conf.get("engine_mode","dineof"))
PY
)
  if [[ "$EMODE" == "dineof" ]]; then
    for st in pre dineof post_dineof; do run_stage "$st"; done
  elif [[ "$EMODE" == "dincae" ]]; then
    for st in pre dincae post_dincae; do run_stage "$st"; done
  else
    for st in pre dineof dincae post_dineof post_dincae; do run_stage "$st"; done
  fi
else
  python lswtctl.py exec --config "$CONF" --row "$SLURM_ARRAY_TASK_ID" --stage "${STAGE}"
fi

echo ""
echo "[$(date)] Completed ${STAGE} for lake_id=${LAKE_ID}, row=${SLURM_ARRAY_TASK_ID}"
"""
    stage_path.write_text(stage_content)
    stage_path.chmod(0o755)
    print(f"[INFO] Generated stage.slurm: {stage_path}")

# ---------- plan/submit/exec ----------

def do_plan(conf_path:str):
    with open(conf_path) as f: conf = json.load(f)
    grid = _grid(conf)
    print(f"Tasks: {len(grid)}  (rows 0..{len(grid)-1})")
    if not grid:
        print("No tasks. Check dataset_options.custom_lake_ids.", file=sys.stderr); sys.exit(1)
    for i,(lake,ver,a) in enumerate(grid[:5]):
        print(f"  row {i}: lake={lake} version={ver} alpha={a}")
    paths0 = _resolve_paths(conf, grid[0][0], grid[0][2])
    print(f"Proposed run_tag: {paths0['run_tag']}")
    print(f"Run root: {paths0['run_root']}")
    print(f"Logs dir: {paths0['logs_dir']}")
    print(f"Engine mode: {conf.get('engine_mode','dineof')}")

def do_submit(conf_path:str):
    _ensure_stage_slurm()
    with open(conf_path) as f: conf = json.load(f)
    grid = _grid(conf)
    if not grid:
        print("No tasks. Check dataset_options.custom_lake_ids.", file=sys.stderr); sys.exit(1)

    paths0 = _resolve_paths(conf, grid[0][0], grid[0][2])
    logd = paths0["logs_dir"]; pathlib.Path(logd).mkdir(parents=True, exist_ok=True)

    _ensure_dir(paths0["run_root"])
    with open(os.path.join(paths0["run_root"], "manifest.json"), "w") as f:
        json.dump(conf, f, indent=2)
    with open(os.path.join(paths0["run_root"], "README.txt"), "w") as f:
        f.write(f"Run tag: {paths0['run_tag']}\nGenerated: {datetime.now(timezone.utc).isoformat()}\n")

    sub = conf.get("submission", {})
    maxc = str(sub.get("max_concurrent", 50))
    part = sub.get("partition"); qos  = sub.get("qos"); acc  = sub.get("account")
    tim  = sub.get("time", "12:00:00"); mem  = sub.get("mem", "8G")
    per_index = bool(sub.get("per_index_chain", False))

    nmax = len(grid) - 1
    base = ["sbatch", "--parsable",
            f"--array=0-{nmax}%{maxc}",
            f"--time={tim}", f"--mem={mem}",
            "--output", "/dev/null",
            "--error", "/dev/null"]
    if part: base += ["-p", part]
    if qos:  base += ["--qos", qos]
    if acc:  base += ["-A", acc]

    here = str(pathlib.Path(__file__).parent)

    def submit_stage(stage, dep=None, name=None):
        env = os.environ.copy()
        env.update({"CONF": os.path.abspath(conf_path), "STAGE": stage, "LOGS_DIR": logd})
        cmd = base + ["--job-name", name or f"lswt_{stage}"]
        if dep: cmd += ["--dependency", dep]
        cmd += ["stage.slurm"]
        job = subprocess.check_output(cmd, env=env, cwd=here).decode().strip()
        return job

    emode = conf.get("engine_mode", "dineof").lower()
    if per_index:
        chain_job = submit_stage("chain", dep=None, name="lswt_chain")
        print(f"Submitted (per-index inline chain): chain={chain_job}  [mode={emode}]")
    else:
        # Stage-wide chaining
        pre_job  = submit_stage("pre", dep=None, name="lswt_pre")
        if emode == "dineof":
            d_job = submit_stage("dineof", dep=f"afterok:{pre_job}", name="lswt_dineof")
            p_job = submit_stage("post_dineof", dep=f"afterok:{d_job}", name="lswt_post_dineof")
            print(f"Submitted: pre={pre_job} → dineof={d_job} → post_dineof={p_job}")
        elif emode == "dincae":
            c_job = submit_stage("dincae", dep=f"afterok:{pre_job}", name="lswt_dincae")
            p_job = submit_stage("post_dincae", dep=f"afterok:{c_job}", name="lswt_post_dincae")
            print(f"Submitted: pre={pre_job} → dincae={c_job} → post_dincae={p_job}")
        else:
            # both: run dineof and dincae after pre (in parallel), then their posts
            d_job = submit_stage("dineof", dep=f"afterok:{pre_job}", name="lswt_dineof")
            c_job = submit_stage("dincae", dep=f"afterok:{pre_job}", name="lswt_dincae")
            pd_job = submit_stage("post_dineof", dep=f"afterok:{d_job}", name="lswt_post_dineof")
            pc_job = submit_stage("post_dincae", dep=f"afterok:{c_job}", name="lswt_post_dincae")
            print(f"Submitted: pre={pre_job} → dineof={d_job} & dincae={c_job} → post_dineof={pd_job} & post_dincae={pc_job}")

def _env_for(conf: dict, stage: str) -> str:
    envs = conf.get("env", {})
    # map post variants
    if stage == "post_dineof":
        e = envs.get("post_dineof", envs.get("post", {}))
    elif stage == "post_dincae":
        e = envs.get("post_dincae", envs.get("post", {}))
    else:
        e = envs.get(stage, envs.get("pre", {}))
    activate = e.get("activate", "")
    module_load = e.get("module_load", "")
    return " && ".join([s for s in [module_load, activate] if s])

def do_exec(conf_path:str, row:int, stage:str):
    with open(conf_path) as f: conf = json.load(f)
    grid = _grid(conf)
    if not (0 <= row < len(grid)):
        print(f"Row {row} out of bounds 0..{len(grid)-1}", file=sys.stderr); sys.exit(1)
    lake_id, version, alpha = grid[row]

    paths = _resolve_paths(conf, lake_id, alpha)
    execs = conf.get("executables", {})
    stage_env = _env_for(conf, stage)

    behavior = conf.get("behavior", {})
    keep_temps = bool(behavior.get("keep_temps", False))
    idempotent = bool(behavior.get("idempotent", True))

    # --- PRE ---
    def run_pre():
        pre_cli = execs.get("pre_cli", "dineof_preprocessor")
        tmpdir = tempfile.mkdtemp(prefix="preconf_")
        pre_json = os.path.join(tmpdir, f"lake_{lake_id}.json")

        var_name = conf.get("variables", {}).get("lswt_var", "lake_surface_water_temperature")
        mask_var = conf.get("variables", {}).get("mask_var", "lakeid")
        time_var = conf.get("variables", {}).get("time_var", "time")

        pre_conf = {
            "lake_id": lake_id, "version": version, "alpha": alpha,
            "input_file": paths["lake_ts"], "input_var": var_name,
            "mask_var": mask_var, "time_var": time_var,
            "output_dir": paths["prepared_dir"], "output_file": paths["prepared_nc"],
            "output": paths["prepared_nc"],
            **conf.get("preprocessing_options", {})
        }
        _ensure_test_id(pre_conf, paths["run_tag"])

        _ensure_dir(paths["prepared_dir"])
        pathlib.Path(pre_json).write_text(json.dumps(pre_conf, indent=2))
        
        # Save persistent copy for provenance/reproducibility
        persistent_config = os.path.join(paths["prepared_dir"], "preprocessing_config.json")
        pathlib.Path(persistent_config).write_text(json.dumps(pre_conf, indent=2))
        print(f"[PRE] Saved config: {persistent_config}", flush=True)

        if idempotent and _idempotent_skip(paths["prepared_nc"], "PRE"):
            if not keep_temps:
                pathlib.Path(pre_json).unlink(missing_ok=True); pathlib.Path(tmpdir).rmdir()
            return

        cmd = f"{pre_cli} --config {pre_json}"
        print("[PRE] Exec:", cmd, flush=True)
        try:
            _bash_exec(cmd, stage_env)
        finally:
            if not keep_temps:
                pathlib.Path(pre_json).unlink(missing_ok=True); pathlib.Path(tmpdir).rmdir()

    # --- DINEOF ---
    def run_dineof():
        dineof_bin = execs.get("dineof_bin", "/home/users/shaerdan/softwares/DINEOF/dineof")
        tmpdir = tempfile.mkdtemp(prefix="dineof_init_")
        init_path = os.path.join(tmpdir, f"lake_{lake_id}.init")
        var_name = conf.get("variables", {}).get("lswt_var", "lake_surface_water_temperature")
        mask_var = conf.get("variables", {}).get("mask_var", "lakeid")
        time_var = conf.get("variables", {}).get("time_var", "time")
        dp = conf.get("dineof_parameters", {})
        nev   = dp.get("nev", 50); neini = dp.get("neini", 1); ncv   = dp.get("ncv", nev+7)
        tol   = dp.get("tol", 1e-8); nitemax = dp.get("nitemax", 300); toliter = dp.get("toliter", 1e-3)
        rec   = dp.get("rec", 1); eof   = dp.get("eof", 1); norm  = dp.get("norm", 0); numit = dp.get("numit", 3)
        seed  = dp.get("seed", 243435)
        use_custom_cv = dp.get("use_custom_cv", False)  # NEW: option to use generated cv_pairs.nc

        _ensure_dir(paths["dineof_dir"])
        
        # Check if custom CV file exists (from preprocessing stage)
        cv_pairs_nc = os.path.join(paths["prepared_dir"], "cv_pairs.nc")
        use_cv_file = use_custom_cv and os.path.exists(cv_pairs_nc)
        
        if use_cv_file:
            print(f"[DINEOF] Using custom CV file: {cv_pairs_nc}", flush=True)
        else:
            if use_custom_cv:
                print(f"[DINEOF] Warning: use_custom_cv=true but {cv_pairs_nc} not found. Using DINEOF built-in CV.", flush=True)
            else:
                print(f"[DINEOF] Using DINEOF built-in random CV method (ncv={ncv})", flush=True)
        
        init_txt = f"""! Auto-generated by lswtctl (ephemeral)
data = ['{paths["prepared_nc"]}#{var_name}']
mask = ['{paths["prepared_nc"]}#{mask_var}']
time = '{paths["prepared_nc"]}#{time_var}'
alpha = {alpha}
numit = {numit}

nev = {nev}
neini = {neini}
ncv = {ncv}
tol = {tol}
nitemax = {nitemax}
toliter = {toliter}
rec = {rec}
eof = {eof}
norm = {norm}

Output = '{paths["dineof_dir"]}/'
results = ['{paths["results_nc_dineof"]}#temp_filled']

seed = {seed}

EOF.U = ['{paths["dineof_dir"]}/eof.nc#Usst']
EOF.V = '{paths["dineof_dir"]}/eof.nc#V'
EOF.Sigma = '{paths["dineof_dir"]}/eof.nc#Sigma'
"""
        
        # Add custom CV (clouds) 
        if use_cv_file:
            init_txt += f"\nclouds = '{cv_pairs_nc}#cv_pairs'\n"
        
        pathlib.Path(init_path).write_text(init_txt)
        
        # Save persistent copy for provenance/reproducibility
        persistent_init = os.path.join(paths["dineof_dir"], "dineof.init")
        pathlib.Path(persistent_init).write_text(init_txt)
        print(f"[DINEOF] Saved init file: {persistent_init}", flush=True)

        if idempotent and _idempotent_skip(paths["results_nc_dineof"], "DINEOF"):
            if not keep_temps:
                pathlib.Path(init_path).unlink(missing_ok=True); pathlib.Path(tmpdir).rmdir()
            return

        cmd = f"{dineof_bin} {init_path}"
        print("[DINEOF] Exec:", cmd, flush=True)
        try:
            _bash_exec(cmd, stage_env)
        finally:
            if not keep_temps:
                pathlib.Path(init_path).unlink(missing_ok=True); pathlib.Path(tmpdir).rmdir()

    # --- DINCAE ---
    def run_dincae():
        # Build DINCAE config from JSON (top-level 'dincae' block if present, else deduce from existing keys)
        dcfg = {
            "lake_id": lake_id,
            "epoch": conf.get("dincae", {}).get("epoch", "1981-01-01T12:00:00Z"),
            "var_name": conf.get("variables", {}).get("lswt_var", "lake_surface_water_temperature"),
            "crop": {"buffer_pixels": conf.get("dincae", {}).get("crop", {}).get("buffer_pixels", 2)},
            "cv": conf.get("dincae", {}).get("cv", {"cv_fraction":0.1, "random_seed":1234, "use_cv": True}),
            "train": conf.get("dincae", {}).get("train", {
                "epochs":300,"batch_size":32,"ntime_win":0,"learning_rate":1e-4,"enc_levels":3,
                "obs_err_std":0.2,"save_epochs_interval":10,"use_gpu":True
            }),
            "runner": conf.get("dincae", {}).get("runner", {
                "mode":"local","julia_exe":"julia","script":"run_dincae.jl","julia_project":True,"skip_existing":True
            }),
            "slurm": conf.get("dincae", {}).get("slurm", {}),
            "post": conf.get("dincae", {}).get("post", {"write_merged": False})
        }
        # >>> PASS STAGE ENV TO DINCAE SUBJOB <<<
        dcfg["env"] = conf.get("env", {})

        # Intermediates live under dincae_dir
        _ensure_dir(paths["dincae_dir"])
        
        # Save persistent copy for provenance/reproducibility
        persistent_dincae_config = os.path.join(paths["dincae_dir"], "dincae_config.json")
        pathlib.Path(persistent_dincae_config).write_text(json.dumps(dcfg, indent=2))
        print(f"[DINCAE] Saved config: {persistent_dincae_config}", flush=True)
        
        prepared = PreparedNC(pathlib.Path(paths["prepared_nc"]))
        arts = dincae_build_inputs(prepared, pathlib.Path(paths["dincae_dir"]), dcfg)
        arts = dincae_run(dcfg, arts)
        # Write a DINEOF-shaped product to dincae_results, then the post stage reads that
        shaped_nc = pathlib.Path(paths["dincae_dir"]) / "dincae_results.nc"
        out = dincae_write_out(
            arts=arts,
            prepared=prepared,
            post_dir=pathlib.Path(paths["dincae_dir"]),  # temporary sink; we'll copy to results path
            final_front_name="__tmp_dincae_for_post__",  # temporary name
            cfg=dcfg
        )
        # Move the shaped output into results_nc_dincae for post stage, and remove temp names
        tmp_output = out["output_nc"]
        if tmp_output and pathlib.Path(tmp_output).exists():
            shutil.copy2(tmp_output, paths["results_nc_dincae"])

    # --- POST (DINEOF/DINCAE separated) ---
    def run_post_dineof():
        post_cli = execs.get("post_cli", "dineof_postprocessor")
        _ensure_dir(paths["post_dir"]); _ensure_dir(paths["html_dir"])

        if idempotent and _idempotent_skip(paths["post_dineof"], "POST_DINEOF"):
            return

        cmd = (
            f"{post_cli} "
            f"--lake-path {paths['lake_ts']} "
            f"--dineof-input-path {paths['prepared_nc']} "
            f"--dineof-output-path {paths['results_nc_dineof']} "
            f"--output-path {paths['post_dineof']} "
            f"--output-html-folder {paths['html_dir']} "
            f"--config-file {os.path.abspath(conf_path)} "
            f"--climatology-file {paths['clim_nc']} "
            f"--units celsius"
        )
        print("[POST_DINEOF] Exec:", cmd, flush=True)
        _bash_exec(cmd, stage_env)

    def run_post_dincae():
        post_cli = execs.get("post_cli", "dineof_postprocessor")
        _ensure_dir(paths["post_dir"]); _ensure_dir(paths["html_dir"])

        if idempotent and _idempotent_skip(paths["post_dincae"], "POST_DINCAE"):
            return

        cmd = (
            f"{post_cli} "
            f"--lake-path {paths['lake_ts']} "
            f"--dineof-input-path {paths['prepared_nc']} "
            f"--dineof-output-path {paths['results_nc_dincae']} "
            f"--output-path {paths['post_dincae']} "
            f"--config-file {os.path.abspath(conf_path)} "
            f"--climatology-file {paths['clim_nc']} "
            f"--units celsius "
            f"--no-eof-filter "      
            f"--no-eof-interp "      
            f"--no-eof-meta "       
            f"--no-log-meta"             
        )
        print("[POST_DINCAE] Exec:", cmd, flush=True)
        _bash_exec(cmd, stage_env)

    # --- dispatch ---
    if stage == "pre":
        run_pre()
    elif stage == "dineof":
        run_dineof()
    elif stage == "dincae":
        run_dincae()
    elif stage == "post_dineof":
        run_post_dineof()
    elif stage == "post_dincae":
        run_post_dincae()
    elif stage == "chain":
        emode = conf.get("engine_mode","dineof").lower()
        if emode == "dineof":
            run_pre(); run_dineof(); run_post_dineof()
        elif emode == "dincae":
            run_pre(); run_dincae(); run_post_dincae()
        else:
            run_pre(); run_dineof(); run_dincae(); run_post_dineof(); run_post_dincae()
    else:
        print(f"Unknown stage: {stage}", file=sys.stderr); sys.exit(2)

# ---------- helper to print paths ----------

def do_paths(conf_path: str, row: int):
    with open(conf_path) as f: conf = json.load(f)
    grid = _grid(conf)
    if not (0 <= row < len(grid)):
        print(f"# row {row} OOB", file=sys.stderr); sys.exit(1)
    lake_id, _, alpha = grid[row]
    P = _resolve_paths(conf, lake_id, alpha)
    for k in ("prepared_nc","results_nc_dineof","results_nc_dincae","post_dineof","post_dincae",
              "lake_ts","clim_nc","dineof_dir","dincae_dir","prepared_dir","post_dir"):
        print(f'{k}={P[k]}')

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("plan");   p1.add_argument("config")
    p2 = sub.add_parser("submit"); p2.add_argument("config")
    p3 = sub.add_parser("exec")
    p3.add_argument("--config", required=True)
    p3.add_argument("--row", required=True, type=int)
    p3.add_argument("--stage", required=True, choices=["pre","dineof","dincae","post_dineof","post_dincae","chain"])
    p4 = sub.add_parser("paths")
    p4.add_argument("--config", required=True)
    p4.add_argument("--row", required=True, type=int)

    args = ap.parse_args()
    if args.cmd == "plan":
        do_plan(args.config)
    elif args.cmd == "submit":
        do_submit(args.config)
    elif args.cmd == "exec":
        do_exec(args.config, args.row, args.stage)
    elif args.cmd == "paths":
        do_paths(args.config, args.row)
    else:
        print("Unknown command", file=sys.stderr); sys.exit(2)

if __name__ == "__main__":
    main()