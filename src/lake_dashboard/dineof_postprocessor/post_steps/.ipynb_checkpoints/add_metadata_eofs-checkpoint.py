# post_steps/add_metadata_eofs.py
from __future__ import annotations

import os
import xarray as xr
from typing import Optional
from .base import PostProcessingStep, PostContext

class AddEOFsMetadataStep(PostProcessingStep):
    """
    Record number of EOFs into attrs by reading eofs.nc (or eof.nc) in Output_test_* folder.
    """

    def should_apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> bool:
        return ds is not None

    def apply(self, ctx: PostContext, ds: Optional[xr.Dataset]) -> xr.Dataset:
        assert ds is not None
        # Locate Output_test_* sibling
        recon_dir = os.path.dirname(os.path.dirname(ctx.output_path))
        postproc_folder = os.path.basename(os.path.dirname(ctx.output_path))
        output_folder = postproc_folder.replace("postprocessed_lake_", "Output_test_")
        eofs_path = os.path.join(recon_dir, output_folder, "eofs.nc")

        if not os.path.isfile(eofs_path):
            alt = os.path.join(recon_dir, output_folder, "eof.nc")
            if os.path.isfile(alt):
                eofs_path = alt
            else:
                print(f"[AddEOFsMetadata] EOFs file not found: {eofs_path} (or {alt})")
                return ds

        try:
            with xr.open_dataset(eofs_path) as e:
                if "eofs" in e.dims:
                    num = int(e.dims["eofs"])
                elif "eof" in e.dims:
                    num = int(e.dims["eof"])
                elif "mode" in e.dims:
                    num = int(e.dims["mode"])
                else:
                    print(f"[AddEOFsMetadata] Unknown EOF dim scheme in {eofs_path}; skipping.")
                    return ds
                ds.attrs["num_eofs"] = num
                ds.attrs["eofs_file"] = eofs_path
        except Exception as ex:
            print(f"[AddEOFsMetadata] Failed to read EOFs file: {ex}")
        return ds
