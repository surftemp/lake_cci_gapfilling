# post_steps/add_metadata_init.py
from __future__ import annotations

import os
import re
import xarray as xr
from typing import Optional, Dict
from .base import PostProcessingStep, PostContext


class AddInitMetadataStep(PostProcessingStep):
    """
    Locate the .init used for DINEOF and add its parameters as attributes.

    Strategy:
      1) Walk up from output_path and extract suffix "..._alpha_<val>"
      2) Find .init in /home/users/shaerdan/lake_dashboard/generated_init_files_<suffix>/lake_<lake_id>.init
      3) Parse key params and attach as attrs.
    """

    INIT_ROOT = "/home/users/shaerdan/lake_dashboard"

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None and "temp_filled" in ds and ctx.lake_id is not None and ctx.lake_id >= 0

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None

        suffix = self._derive_suffix_from_output_path(ctx.output_path)
        if not suffix:
            print("[AddInitMetadata] Could not derive suffix from output_path; skipping.")
            return ds

        init_dir = os.path.join(self.INIT_ROOT, f"generated_init_files_{suffix}")
        init_path = os.path.join(init_dir, f"lake_{ctx.lake_id}.init")
        if not os.path.isfile(init_path):
            print(f"[AddInitMetadata] .init file not found: {init_path}")
            return ds

        params = self._parse_init(init_path)
        for k, v in params.items():
            ds.attrs[f"dineof_{k}"] = v
        ds.attrs["dineof_init_file"] = init_path
        return ds

    @staticmethod
    def _derive_suffix_from_output_path(output_path: str) -> Optional[str]:
        # climb up directories until we match "*_recon_<...>_alpha_<num>"
        cur = os.path.dirname(output_path)
        pat = re.compile(r'^(?:temperature|anomaly)_recon_(.+)_alpha_([0-9]*\.?[0-9]+)$', re.IGNORECASE)
        while True:
            base = os.path.basename(cur)
            m = pat.match(base)
            if m:
                core, alpha_val = m.group(1), m.group(2)
                return f"{core}_alpha_{alpha_val}"
            parent = os.path.dirname(cur)
            if parent == cur:
                return None
            cur = parent

    @staticmethod
    def _parse_init(init_path: str) -> Dict[str, float | int]:
        params = {}
        with open(init_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith(("!", "#")) or "=" not in s:
                    continue
                k, v = [x.strip() for x in s.split("=", 1)]
                if k not in {
                    "alpha", "numit", "nev", "neini", "ncv", "tol",
                    "nitemax", "toliter", "rec", "eof", "norm"
                }:
                    continue
                if any(c in v for c in (".", "e", "E")):
                    try:
                        params[k] = float(v)
                    except Exception:
                        pass
                else:
                    try:
                        params[k] = int(v)
                    except Exception:
                        pass
        return params
