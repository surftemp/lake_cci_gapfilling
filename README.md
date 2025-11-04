# Lake CCI LSWT Gap-Filling Pipeline

End-to-end orchestration to **prepare**, **reconstruct**, and **post-process** Lake Surface Water Temperature (LSWT) using:
- **DINEOF** (EOF-based) and/or
- **DINCAE** (neural autoencoder; integrated via a thin adaptor layer).

A single controller (`lswtctl.py`) reads one JSON file to run **DINEOF**, **DINCAE**, or **BOTH** and writes final NetCDFs with strict naming rules.

---

## Repository layout

```
lake_cci_gapfilling/
├─ config/
│  └─ experiment_settings.json
│
├─ orchestration/
│  ├─ lswtctl.py
│  └─ stage.slurm
│
├─ src/
│  ├─ processors/
│  │  ├─ preprocessor/
│  │  │  └─ lswt_processing/
│  │  │     ├─ ... (filters, climatology, detrending, etc.)
│  │  └─ postprocessor/
│  │     └─ post_steps/
│  │        ├─ ... (filter_eofs, reconstruct_from_eofs, qa_plots, etc.)
│  │
│  ├─ dincae_arm/                  # DINCAE adaptor layer (see its README)
│  │  ├─ __init__.py
│  │  ├─ contracts.py
│  │  ├─ dincae_adapter_in.py
│  │  ├─ dincae_runner.py
│  │  └─ dincae_adapter_out.py
│  │
│  └─ post_analyzer/
│
└─ README.md
```

---

## Overview

The unified controller can execute one or both reconstruction engines.  
Each engine writes its own intermediate folder (`dineof/`, `dincae/`) and the post stage writes identical‑front filenames:

```
.../post/{lake_id9}/{alpha_slug}/LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-filled_fine_dineof.nc
.../post/{lake_id9}/{alpha_slug}/LAKE{lake_id9}-CCI-L3S-LSWT-CDR-4.5-filled_fine_dincae.nc
```

---


## Installation
```
Create an environment using:
    mamba create -n lake_cci_gapfilling python=3.10 -y
    mamba activate lake_cci_gapfilling
    mamba install xarray netcdf4 bokeh selenium firefox geckodriver -y

Install the tools from this repo using:
    conda activate lake_cci_gapfilling
    git clone git@github.com:surftemp/lake_dashboard.git
    cd lake_dashboard
    pip install -e .
```

## Configuration

Everything is controlled by `config/experiment_settings.json`.

Important keys:

| Key | Description |
|-----|--------------|
| `engine_mode` | `"dineof"`, `"dincae"`, or `"both"` |
| `paths.*` | Directory templates for prepared, dineof, dincae, post, etc. |
| `dineof_parameters` | Standard DINEOF settings |
| `dincae.*` | DINCAE hyperparameters, Julia runner options, Slurm overrides |
| `submission.*` | Job array size, partition, QoS, dependencies |

---

## Usage

```bash
# 1. Plan
python orchestration/lswtctl.py plan config/experiment_settings.json

# 2. Submit Slurm jobs
python orchestration/lswtctl.py submit config/experiment_settings.json

# 3. Run one row manually
python orchestration/lswtctl.py exec --config config/experiment_settings.json --row 0 --stage chain
```

The orchestrator automatically expands the lake grid, builds run tags, and submits either:
- single per‑index chains (`pre → dineof|dincae → post_*`) or  
- stage‑wide arrays with dependencies (`pre → dineof/dincae → post_*`).

---

## Output structure

```
prepared/{lake_id9}/prepared.nc
dineof/{lake_id9}/{alpha_slug}/dineof_results.nc
dincae/{lake_id9}/{alpha_slug}/dincae_results.nc
post/{lake_id9}/{alpha_slug}/LAKE{lake_id9}-..._dineof.nc
post/{lake_id9}/{alpha_slug}/LAKE{lake_id9}-..._dincae.nc
```

---

## DINCAE adaptor summary

The `src/dincae_arm/` module lets DINCAE behave like DINEOF:

1. `dincae_adapter_in.py` → convert `prepared.nc` → DINCAE tensors  
2. `dincae_runner.py` → run Julia DINCAE locally or via Slurm  
3. `dincae_adapter_out.py` → rebuild full‑grid prediction; align to DINEOF shape

Post‑processing then runs unmodified.

See [`src/dincae_arm/README.md`](src/dincae_arm/README.md) for details.

---

## Citation

Please cite the original **DINEOF** and **DINCAE** works and this pipeline when publishing derived products.
