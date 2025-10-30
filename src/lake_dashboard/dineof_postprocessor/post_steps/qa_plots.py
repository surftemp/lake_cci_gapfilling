# post_steps/qa_plots.py
from __future__ import annotations

import os
from typing import Optional
import xarray as xr
from .base import PostProcessingStep, PostContext


class QAPlotsStep(PostProcessingStep):
    """
    Optional QA plots: re-run EOF reconstruction and export figures.

    Uses:
      - NewReconstructor and PlotExporter from your existing modules.
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and bool(ctx.output_html_folder)

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None
        try:
            # Lazy imports so module is optional
            from ..reconstruct_eofs import NewReconstructor
            from ..batch_eof_plot import PlotExporter
        except Exception as e:
            print(f"[QAPlots] Plot modules not available ({e}); skipping QA plots.")
            return ds

        recon_dir = os.path.dirname(ctx.dineof_output_path)
        out_html = ctx.output_html_folder
        os.makedirs(out_html, exist_ok=True)

        try:
            NewReconstructor(ctx.dineof_input_path, "lakeid", recon_dir, "eofs.nc").run()
            # Lake id & test id for titles (if available in attrs)
            lake_id = ds.attrs.get("lake_id", None)
            test_id = ds.attrs.get("test_id", None)
            PlotExporter(out_html, recon_dir, "eofs.nc", lake_id, test_id, ds.attrs).run()
        except Exception as e:
            print(f"[QAPlots] Failed to generate QA plots: {e}")

        return ds
