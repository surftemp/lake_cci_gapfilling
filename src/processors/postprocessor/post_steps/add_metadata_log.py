# post_steps/add_metadata_log.py
from __future__ import annotations

import os
import re
from glob import glob
import xarray as xr
from typing import Optional
from .base import PostProcessingStep, PostContext


class AddDineofLogMetadataStep(PostProcessingStep):
    """
    Parse the latest *.out in Output_test_* folder to record CV error / missing stats.
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        recon_dir = os.path.dirname(os.path.dirname(ctx.output_path))
        postproc_folder = os.path.basename(os.path.dirname(ctx.output_path))
        output_folder = postproc_folder.replace("postprocessed_lake_", "Output_test_")
        out_dir = os.path.join(recon_dir, output_folder)

        candidates = glob(os.path.join(out_dir, "*.out"))
        if not candidates:
            print(f"[AddDineofLogMetadata] No .out logs in {out_dir}")
            return ds
        log_path = max(candidates, key=os.path.getmtime)

        try:
            txt = open(log_path, "r", errors="ignore").read()
        except Exception as e:
            print(f"[AddDineofLogMetadata] Failed reading log: {e}")
            return ds

        m_missing = re.search(
            r"Missing\s+data:\s*(\d+)\s+out\s+of\s+(\d+)\s*\(\s*([0-9.]+)\s*%\s*\)",
            txt, flags=re.IGNORECASE
        )
        if m_missing:
            ds.attrs["dineof_missing_count"] = int(m_missing.group(1))
            ds.attrs["dineof_total_count"] = int(m_missing.group(2))
            ds.attrs["dineof_missing_percent"] = float(m_missing.group(3))

        m_err = re.search(
            r"expected\s+error\s+calculated\s+by\s+cross-validation\s+([0-9.]+)",
            txt, flags=re.IGNORECASE
        )
        if m_err:
            ds.attrs["dineof_cv_expected_error"] = float(m_err.group(1))

        ds.attrs["dineof_log_file"] = log_path
        return ds
