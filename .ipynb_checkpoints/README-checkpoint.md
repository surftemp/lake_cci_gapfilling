# lake_cci_gapfilling
gap filling lake surface water temperature 


lake_cci_gapfilling/
├─ config/
│  └─ experiment_settings.json            # CENTRAL CONTROL (see D)
│
├─ orchestration/
│  ├─ lswtctl.py                          # EDIT: read config/run.mode, stage chains
│  └─ stage.slurm
│
├─ src/
│  ├─ processors/                         # 
│  │  ├─ preprocessor/                    # 
│  │  │  ├─ convert_data.py 
│  │  │  ├─ __init__.py
│  │  │  ├─ lswt_processing/
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ base.py                    # existing
│  │  │  │  ├─ config.py                  # existing
│  │  │  │  ├─ data_loading.py            # existing
│  │  │  │  ├─ quality_filters.py         # existing
│  │  │  │  ├─ frame_filters.py           # existing
│  │  │  │  ├─ spatial_filters.py         # existing
│  │  │  │  ├─ ice_filter.py              # existing
│  │  │  │  ├─ climatology.py             # existing
│  │  │  │  ├─ detrending.py              # existing
│  │  │  │  ├─ finalization.py            # existing
│  │  │  │  └─ stats.py                   # existing
│  │  │  └─ __init__.py
│  │  │
│  │  └─ postprocessor/                   # 
│  │  │  ├─ post_process.py
│  │  │  ├─ recosntruct_eofs.py
│  │  │  ├─ batch_eof_plots.py
│  │     ├─ post_steps/
│  │     │  ├─ __init__.py
│  │     │  ├─ base.py                    # existing
│  │     │  ├─ merge_outputs.py           # existing
│  │     │  ├─ copy_aux_flags.py          # existing
│  │     │  ├─ add_back_trend.py          # existing
│  │     │  ├─ add_back_climatology.py    # existing
│  │     │  ├─ add_metadata_init.py       # existing
│  │     │  ├─ add_metadata_eofs.py       # existing
│  │     │  ├─ qa_plots.py                # existing
│  │     │  ├─ filter_eofs.py             # existing
│  │     │  ├─ interpolate_temporal_eofs.py # existing
│  │     │  └─ reconstruct_from_eofs.py   # existing
│  │     ├─ post_process.py               # existing
│  │     └─ __init__.py
│  │
│  ├─ gapflow/                            # NEW: adapters & runners for DINCAE
│  │  ├─ __init__.py
│  │  ├─ contracts.py                     # NEW: tiny path/schema helpers
│  │  ├─ dincae_adapter_in.py             # NEW: prepared.nc → DINCAE inputs
│  │  ├─ dincae_adapter_out.py            # NEW: DINCAE outputs → model_output.nc
│  │  ├─ dincae_runner.py                 # NEW: SLURM/local runner
│  │  └─ run_dincae_once.py               # NEW: one-lake entrypoint for lswtctl
│  │
│  └─ dineof_bridge/                      # NEW: normalize DINEOF naming (thin)
│     ├─ __init__.py
│     └─ to_model_output.py               # NEW: output.nc → model_output.nc
│
└─ README.md
