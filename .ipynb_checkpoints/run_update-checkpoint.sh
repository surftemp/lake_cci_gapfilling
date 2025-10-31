#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="/home/users/shaerdan/lake_cci_gapfilling/src/dincae_arm"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${TARGET_DIR}"

backup_if_exists() {
  local f="$1"
  if [[ -f "${f}" ]]; then
    cp -a "${f}" "${f}.bak.${STAMP}"
    echo "Backed up ${f} -> ${f}.bak.${STAMP}"
  fi
}

# ----- __init__.py -----
backup_if_exists "${TARGET_DIR}/__init__.py"
cat > "${TARGET_DIR}/__init__.py" << 'PY'
from .contracts import PreparedNC, DincaeArtifacts
from .dincae_adapter_in import build_inputs
from .dincae_runner import run as run_dincae
from .dincae_adapter_out import write_dineof_shaped_outputs

__all__ = [
    "PreparedNC",
    "DincaeArtifacts",
    "build_inputs",
    "run_dincae",
    "write_dineof_shaped_outputs",
]
PY

# ----- contracts.py -----
backup_if_exists "${TARGET_DIR}/contracts.py"
cat > "${TARGET_DIR}/contracts.py" << 'PY'
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class PreparedNC:
    """Original pipeline prepared file (usually prepared.nc)."""
    path: Path

@dataclass
class DincaeArtifacts:
    """Manifest for DINCAE adaptor steps."""
    dincae_dir: Path  # e.g. {run_root}/dincae/{lake_id9}/{alpha_slug}
    prepared_datetime: Path                 # prepared_datetime.nc
    prepared_cropped: Path                  # prepared_datetime_cropped.nc
    prepared_cropped_cv: Path               # prepared_datetime_cropped_add_clouds.nc
    prepared_cropped_clean: Path            # prepared_datetime_cropped_add_clouds.clean.nc
    pred_path: Optional[Path] = None        # data-avg.nc (cropped)
    pred_full_path: Optional[Path] = None   # data-avg-full.nc (full grid/time)
PY

# ----- dincae_adapter_in.py -----
backup_if_exists "${TARGET_DIR}/dincae_adapter_in.py"
cat > "${TARGET_DIR}/dincae_adapter_in.py" << 'PY'
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import xarray as xr
from .contracts import PreparedNC, DincaeArtifacts

def _as_datetime64_from_int(time_da: xr.DataArray, epoch: str) -> xr.DataArray:
    if np.issubdtype(time_da.dtype, np.datetime64):
        return time_da
    if not (np.issubdtype(time_da.dtype, np.integer) or np.issubdtype(time_da.dtype, np.floating)):
        return time_da
    base = np.datetime64(epoch.replace("Z", ""), "ns")
    vals = np.asarray(time_da.values, dtype=float)
    out = base + (vals * np.timedelta64(1, "D")).astype("timedelta64[ns]")
    return xr.DataArray(out, dims=time_da.dims, coords=time_da.coords, name=time_da.name, attrs=time_da.attrs)

def convert_time(in_nc: Path, out_nc: Path, epoch: str) -> None:
    ds = xr.load_dataset(in_nc)
    if "time" not in ds:
        raise ValueError("No 'time' coordinate in dataset.")
    ds = ds.assign_coords(time=_as_datetime64_from_int(ds["time"], epoch))
    ds.to_netcdf(out_nc)

def _bbox_from_mask(mask2d: np.ndarray, buffer: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        raise ValueError("Lake mask has no valid (==1) pixels.")
    i0, i1 = int(ys.min()), int(ys.max()) + 1
    j0, j1 = int(xs.min()), int(xs.max()) + 1
    i0 = max(0, i0 - buffer); j0 = max(0, j0 - buffer)
    i1 = min(mask2d.shape[0], i1 + buffer)
    j1 = min(mask2d.shape[1], j1 + buffer)
    return i0, i1, j0, j1

def crop_to_mask(in_nc: Path, out_nc: Path, buffer: int) -> None:
    ds = xr.load_dataset(in_nc)
    if "lakeid" not in ds:
        raise ValueError("Expected variable 'lakeid' to locate the lake mask (1==water).")
    lakeid = ds["lakeid"].load()
    if lakeid.ndim != 2:
        raise ValueError("Expected lakeid(lat,lon) 2D mask.")
    mask = (lakeid == 1).values
    i0, i1, j0, j1 = _bbox_from_mask(mask, buffer)
    lat_name, lon_name = "lat", "lon"
    if lat_name not in ds.dims or lon_name not in ds.dims:
        dims = list(ds.dims)
        if len(dims) >= 2: lat_name, lon_name = dims[-2], dims[-1]
        else: raise ValueError("Cannot determine (lat, lon) dimension names.")
    ds_crop = ds.isel({lat_name: slice(i0, i1), lon_name: slice(j0, j1)}).copy()
    ds_crop.attrs.update({
        "crop_i0": i0, "crop_i1": i1,
        "crop_j0": j0, "crop_j1": j1,
        "crop_buffer_pixels": int(buffer),
    })
    ds_crop.to_netcdf(out_nc)

def add_cv_clouds(in_nc: Path, out_cv_nc: Path, out_clean_nc: Path,
                  cv_fraction: float = 0.1,
                  random_seed: int | None = 1234,
                  variable_name: str = "lake_surface_water_temperature") -> None:
    ds = xr.load_dataset(in_nc)
    ds.to_netcdf(out_clean_nc)
    if "time" not in ds: raise ValueError("Dataset must have time dimension for CV masking.")
    if variable_name not in ds: raise ValueError(f"Variable '{variable_name}' not found for CV masking.")
    rng = np.random.default_rng(random_seed)
    ntime = ds.sizes["time"]
    k = max(1, int(round(cv_fraction * ntime)))
    sel_idx = np.sort(rng.choice(ntime, size=k, replace=False))
    var = ds[variable_name].copy()
    if "lakeid" not in ds: raise ValueError("Expected 'lakeid' to identify lake pixels.")
    lake_mask = (ds["lakeid"] == 1)
    var_mask = xr.zeros_like(var, dtype=bool)
    var_mask.loc[dict(time=ds["time"].isel(time=sel_idx))] = True
    lm = lake_mask
    for d in var.dims:
        if d not in lm.dims and d != "time":
            lm = lm.expand_dims({d: var.sizes[d]}) if d in ("lat", "lon") else lm
    masked = var.where(~(var_mask & lm), other=np.nan)
    ds_out = ds.copy()
    ds_out[variable_name] = masked
    ds_out.attrs["cv_masked_frac"] = float(cv_fraction)
    ds_out.attrs["cv_masked_steps"] = int(k)
    ds_out.to_netcdf(out_cv_nc)

def build_inputs(prepared, dincae_dir: Path, cfg: Dict) -> DincaeArtifacts:
    dincae_dir.mkdir(parents=True, exist_ok=True)
    epoch = cfg.get("epoch", "1981-01-01T12:00:00Z")
    buffer = int(cfg.get("crop", {}).get("buffer_pixels", 2))
    var_name = cfg.get("var_name", "lake_surface_water_temperature")
    p_datetime = dincae_dir / "prepared_datetime.nc"
    convert_time(prepared.path, p_datetime, epoch=epoch)
    p_crop = dincae_dir / "prepared_datetime_cropped.nc"
    crop_to_mask(p_datetime, p_crop, buffer=buffer)
    p_cv = dincae_dir / "prepared_datetime_cropped_add_clouds.nc"
    p_clean = dincae_dir / "prepared_datetime_cropped_add_clouds.clean.nc"
    add_cv_clouds(p_crop, p_cv, p_clean,
        cv_fraction=float(cfg.get("cv", {}).get("cv_fraction", 0.1)),
        random_seed=cfg.get("cv", {}).get("random_seed", 1234),
        variable_name=var_name)
    return DincaeArtifacts(
        dincae_dir=dincae_dir,
        prepared_datetime=p_datetime,
        prepared_cropped=p_crop,
        prepared_cropped_cv=p_cv,
        prepared_cropped_clean=p_clean)
PY

# ----- dincae_runner.py -----
backup_if_exists "${TARGET_DIR}/dincae_runner.py"
cat > "${TARGET_DIR}/dincae_runner.py" << 'PY'
from __future__ import annotations
import os, shlex, subprocess
from pathlib import Path
from typing import Dict
from .contracts import DincaeArtifacts

def _build_julia_cmd(cfg: Dict, arts: DincaeArtifacts) -> list[str]:
    julia = cfg.get("runner", {}).get("julia_exe", "julia")
    script = cfg.get("runner", {}).get("script", "run_dincae.jl")
    use_cv = bool(cfg.get("cv", {}).get("use_cv", True))
    in_path = arts.prepared_cropped_cv if use_cv else arts.prepared_cropped
    args = {
        "--in": str(in_path),
        "--outdir": str(arts.dincae_dir),
        "--epochs": cfg.get("train", {}).get("epochs", 300),
        "--batch": cfg.get("train", {}).get("batch_size", 32),
        "--ntime_win": cfg.get("train", {}).get("ntime_win", 0),
        "--lr": cfg.get("train", {}).get("learning_rate", 1e-4),
        "--enc_levels": cfg.get("train", {}).get("enc_levels", 3),
        "--obs_err_std": cfg.get("train", {}).get("obs_err_std", 0.2),
        "--save_interval": cfg.get("train", {}).get("save_epochs_interval", 10),
        "--use_gpu": int(bool(cfg.get("train", {}).get("use_gpu", True))),
    }
    cmd = [julia]
    if cfg.get("runner", {}).get("julia_project", True):
        cmd += ["--project"]
    cmd += [script]
    for k, v in args.items(): cmd += [k, str(v)]
    return cmd

def run_julia_local(arts: DincaeArtifacts, cfg: Dict) -> None:
    env = os.environ.copy()
    if "CUDA_VISIBLE_DEVICES" in cfg.get("runner", {}):
        env["CUDA_VISIBLE_DEVICES"] = str(cfg["runner"]["CUDA_VISIBLE_DEVICES"])
    if "JULIA_PROJECT" in cfg.get("runner", {}):
        env["JULIA_PROJECT"] = str(cfg["runner"]["JULIA_PROJECT"])
    cmd = _build_julia_cmd(cfg, arts)
    subprocess.check_call(cmd, env=env, cwd=str(arts.dincae_dir))

def submit_slurm_job(arts: DincaeArtifacts, cfg: Dict) -> None:
    slurm = cfg.get("slurm", {})
    lake_id = cfg.get("lake_id", "lake")
    script_path = Path(arts.dincae_dir) / f"run_dincae_{lake_id}.slurm"
    log_out = slurm.get("log_out", f"logs_dincae_{lake_id}.out")
    log_err = slurm.get("log_err", f"logs_dincae_{lake_id}.err")
    cmd = " ".join(shlex.quote(p) for p in _build_julia_cmd(cfg, arts))
    script = f"""#!/bin/bash
#SBATCH -J dincae_{lake_id}
#SBATCH -o {log_out}
#SBATCH -e {log_err}
#SBATCH -p {slurm.get('partition','orchid')}
#SBATCH --gres=gpu:{slurm.get('gpus',1)}
#SBATCH -t {slurm.get('time','24:00:00')}
#SBATCH --mem={slurm.get('mem','128G')}
#SBATCH -c {slurm.get('cpus',4)}
{f"#SBATCH -A {slurm['account']}" if 'account' in slurm else ''}
{f"#SBATCH --qos={slurm['qos']}" if 'qos' in slurm else ''}
cd {arts.dincae_dir}
{cmd}
"""
    script_path.write_text(script)
    subprocess.check_call(["sbatch", str(script_path)])

def run(cfg: Dict, arts: DincaeArtifacts) -> DincaeArtifacts:
    mode = cfg.get("runner", {}).get("mode", "local")
    skip_existing = bool(cfg.get("runner", {}).get("skip_existing", True))
    pred = arts.dincae_dir / "data-avg.nc"
    if skip_existing and pred.exists():
        arts.pred_path = pred
        return arts
    if mode == "local":
        run_julia_local(arts, cfg)
    else:
        submit_slurm_job(arts, cfg)
    if pred.exists():
        arts.pred_path = pred
    return arts
PY

# ----- dincae_adapter_out.py -----
backup_if_exists "${TARGET_DIR}/dincae_adapter_out.py"
cat > "${TARGET_DIR}/dincae_adapter_out.py" << 'PY'
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import xarray as xr
from .contracts import PreparedNC, DincaeArtifacts

def post_map_to_full(pred_nc: Path, cropped_nc: Path, prepared_full_nc: Path, out_full_nc: Path) -> None:
    ds_pred = xr.load_dataset(pred_nc)
    ds_crop = xr.load_dataset(cropped_nc)
    ds_ref = xr.load_dataset(prepared_full_nc)
    try:
        i0 = int(ds_crop.attrs["crop_i0"]); i1 = int(ds_crop.attrs["crop_i1"])
        j0 = int(ds_crop.attrs["crop_j0"]); j1 = int(ds_crop.attrs["crop_j1"])
    except KeyError:
        raise ValueError("Cropped dataset lacks crop indices (crop_i0/1, crop_j0/1).")
    time_name, lat_name, lon_name = "time", "lat", "lon"
    var_name = "lake_surface_water_temperature"
    if var_name not in ds_pred:
        candidates = [v for v in ds_pred.data_vars if ds_pred[v].ndim >= 2]
        if not candidates: raise ValueError("Prediction file has no valid data variable.")
        var_name = candidates[0]
    T, NY, NX = ds_ref.sizes[time_name], ds_ref.sizes[lat_name], ds_ref.sizes[lon_name]
    full_vals = np.full((T, NY, NX), np.nan, dtype=float)
    ds_pred = ds_pred.sortby(time_name); ds_ref = ds_ref.sortby(time_name)
    pred_reindexed = ds_pred[var_name].reindex({time_name: ds_ref[time_name]})
    full_vals[:, i0:i1, j0:j1] = pred_reindexed.values
    da_full = xr.DataArray(
        full_vals, dims=(time_name, lat_name, lon_name),
        coords={time_name: ds_ref[time_name], lat_name: ds_ref[lat_name], lon_name: ds_ref[lon_name]},
        name=var_name)
    out = xr.Dataset({var_name: da_full})
    out[var_name].attrs.update(ds_pred[var_name].attrs)
    out.attrs.update({"source_model": "DINCAE", "mapping": "cropped_to_full", "quality_level": 5})
    out.to_netcdf(out_full_nc)

def to_dineof_shape(full_nc: Path, prepared_full_nc: Path, out_nc: Path, var_name: str) -> None:
    ds_full = xr.load_dataset(full_nc)
    ds_ref = xr.load_dataset(prepared_full_nc)
    if var_name not in ds_full:
        candidates = [v for v in ds_full.data_vars]
        if not candidates: raise ValueError("Full product has no data variable to rename.")
        ds_full = ds_full.rename({candidates[0]: var_name})
    ds_out = ds_full.reindex_like(ds_ref, method=None, copy=True)
    ds_out[var_name].attrs.setdefault("units", ds_ref[var_name].attrs.get("units", "kelvin"))
    ds_out[var_name].attrs.setdefault("long_name", "lake surface water temperature (DINCAE)")
    ds_out.attrs.setdefault("source_model", "DINCAE")
    ds_out.to_netcdf(out_nc)

def make_merged(output_nc: Path, prepared_full_nc: Path, out_merged_nc: Path, var_name: str) -> None:
    ds_pred = xr.load_dataset(output_nc)
    ds_ref = xr.load_dataset(prepared_full_nc)
    if var_name not in ds_pred or var_name not in ds_ref:
        raise ValueError(f"Expected '{var_name}' in both datasets.")
    obs = ds_ref[var_name]
    pred = ds_pred[var_name].reindex_like(obs)
    merged = xr.where(~np.isnan(obs), obs, pred)
    ds_out = xr.Dataset({var_name: merged})
    ds_out[var_name].attrs.update(ds_pred[var_name].attrs)
    ds_out.attrs.update({"merge_rule": "obs_priority_then_dincae", "source_model": "DINCAE"})
    ds_out.to_netcdf(out_merged_nc)

def write_dineof_shaped_outputs(
    arts: DincaeArtifacts,
    prepared: PreparedNC,
    post_dir: Path,
    final_front_name: str,
    cfg: Dict
) -> Dict[str, Optional[Path]]:
    post_dir.mkdir(parents=True, exist_ok=True)
    if arts.pred_path is None or not arts.pred_path.exists():
        raise FileNotFoundError("DINCAE prediction (data-avg.nc) not found.")
    full_nc = arts.dincae_dir / "data-avg-full.nc"
    post_map_to_full(
        pred_nc=arts.pred_path,
        cropped_nc=arts.prepared_cropped,
        prepared_full_nc=prepared.path,
        out_full_nc=full_nc)
    arts.pred_full_path = full_nc
    var_name = cfg.get("var_name", "lake_surface_water_temperature")
    output_nc = post_dir / f"{final_front_name}_dincae.nc"
    to_dineof_shape(full_nc=full_nc, prepared_full_nc=prepared.path, out_nc=output_nc, var_name=var_name)
    merged_nc = None
    if bool(cfg.get("post", {}).get("write_merged", False)):
        merged_nc = post_dir / f"{final_front_name}_merged_dincae.nc"
        make_merged(output_nc=output_nc, prepared_full_nc=prepared.path, out_merged_nc=merged_nc, var_name=var_name)
    for p in [output_nc, merged_nc, full_nc]:
        if p and Path(p).exists():
            ds = xr.load_dataset(p); ds.attrs["model_suffix"] = "_dincae"; ds.to_netcdf(p)
    return {"output_nc": output_nc, "merged_nc": merged_nc, "full_nc": full_nc}
PY

echo "Done. Installed/updated dincae_arm files in ${TARGET_DIR}"
