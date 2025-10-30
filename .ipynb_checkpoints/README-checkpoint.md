lake_cci_gapfilling/
├─ config/
│  └─ experiment_settings.json
│
├─ orchestration/
│  ├─ lswtctl.py                  # EDIT: read run.mode; build chains
│  └─ stage.slurm
│
├─ src/
│  ├─ processors/
│  │  ├─ preprocessor/
│  │  │  ├─ convert_data.py
│  │  │  ├─ __init__.py
│  │  │  └─ lswt_processing/
│  │  │     ├─ __init__.py
│  │  │     ├─ base.py
│  │  │     ├─ config.py
│  │  │     ├─ data_loading.py
│  │  │     ├─ quality_filters.py
│  │  │     ├─ frame_filters.py
│  │  │     ├─ spatial_filters.py
│  │  │     ├─ ice_filter.py
│  │  │     ├─ climatology.py
│  │  │     ├─ detrending.py
│  │  │     ├─ finalization.py
│  │  │     └─ stats.py
│  │  └─ postprocessor/
│  │     ├─ post_process.py
│  │     ├─ recosntruct_eofs.py
│  │     ├─ batch_eof_plots.py
│  │     ├─ post_steps/
│  │     │  ├─ __init__.py
│  │     │  ├─ base.py
│  │     │  ├─ merge_outputs.py
│  │     │  ├─ copy_aux_flags.py
│  │     │  ├─ add_back_trend.py
│  │     │  ├─ add_back_climatology.py
│  │     │  ├─ add_metadata_init.py
│  │     │  ├─ add_metadata_eofs.py
│  │     │  ├─ qa_plots.py
│  │     │  ├─ filter_eofs.py
│  │     │  ├─ interpolate_temporal_eofs.py
│  │     │  └─ reconstruct_from_eofs.py
│  │
│  ├─ gapflow/                    # DINCAE adapters & runner
│  │  ├─ __init__.py
│  │  ├─ contracts.py
│  │  ├─ dincae_adapter_in.py     # prepared.nc → DINCAE tensors
│  │  ├─ dincae_runner.py         # launch DINCAE (SLURM/local)
│  │  └─ dincae_adapter_out.py    # DINCAE → DINEOF-shaped output.nc(/merged.nc)
│  │
│  └─ post_analyzer/              # post analysis tools
│     ├─ __init__.py
│     ├─ plots/
│     └─ stats/
│
└─ README.md
