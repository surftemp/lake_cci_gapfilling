# lswt_processing/dineof_safety.py
from __future__ import annotations
import os, re, shutil
from typing import Iterable, Optional, Tuple, List
import numpy as np
import xarray as xr

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig

# Optional fallback search roots if inference fails (kept minimal and deterministic)
INIT_SEARCH_DEFAULTS: tuple[str, ...] = (
    "/home/users/shaerdan/lake_dashboard",  # where script_gen puts init files
)

class DineofInitSafetyAdjustStep(ProcessingStep):
    """Mandatory post-step: adjust DINEOF nev/ncv to fit min(T,N) with ncv > nev+5."""

    @property
    def name(self) -> str:
        return "DINEOF Init Safety Adjust (auto)"

    def should_apply(self, config: ProcessingConfig) -> bool:
        # Always run at the end; if init cannot be located we just log and return.
        return True

    # ---------- path inference from prepared.nc (matches script_gen.py conventions) ----------
    @staticmethod
    def _infer_init_path_from_output(output_nc: str) -> Optional[str]:
        """
        Example prepared path:
          /.../anomaly_recon_v2<SUFFIX>/dineof_test_<LAKEID>_(fine|coarse)/prepared.nc
        or:
          /.../temperature_recon_v2<SUFFIX>/dineof_test_<LAKEID>_(fine|coarse)/prepared.nc

        We map to:
          /home/users/shaerdan/lake_dashboard/generated_init_files_v2<SUFFIX>/lake_<LAKEID>.init
        or:
          /home/users/shaerdan/lake_dashboard/generated_init_files_v0<SUFFIX>/lake_<LAKEID>.init
        """
        p = os.path.abspath(output_nc)
        m = re.search(
            r"/(anomaly_recon_v2|temperature_recon_v2)([^/]*)/dineof_test_(\d+)_(fine|coarse)/prepared\.nc$",
            p,
        )
        if not m:
            return None
        bucket, suffix, lake_str, _res = m.groups()
        lid = int(lake_str)
        version_tag = "v2" if bucket.startswith("anomaly_recon_v2") else "v0"
        base_subdir = f"generated_init_files_{version_tag}"
        init_dir = f"/home/users/shaerdan/lake_dashboard/{base_subdir}{suffix}/"
        return os.path.join(init_dir, f"lake_{lid}.init")

    # ---------- tiny fallback search (only under the canonical base) ----------
    @staticmethod
    def _find_init_containing_prepared(output_nc: str, roots: Iterable[str]) -> Optional[str]:
        needle = f"data = ['{os.path.abspath(output_nc)}#lake_surface_water_temperature']"
        for root in roots:
            root = os.path.abspath(root)
            if not os.path.isdir(root):
                continue
            for dirpath, _, files in os.walk(root):
                for fn in files:
                    if not fn.endswith(".init"):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        with open(p, "r") as f:
                            txt = f.read()
                        if needle in txt:
                            return p
                    except Exception:
                        pass
        return None

    # ---------- core utilities ----------
    @staticmethod
    def _matrix_dims(ds: xr.Dataset) -> Tuple[int, int]:
        if "time" not in ds.dims:
            raise ValueError("No 'time' dimension after preprocessing.")
        T = int(ds.dims["time"])
        if "lakeid" in ds.variables:
            mask = (ds["lakeid"].data > 0)
        elif "_mask" in ds.attrs:
            mask = np.asarray(ds.attrs["_mask"]).astype(bool)
        else:
            raise ValueError("No lake mask ('lakeid' or '_mask').")
        if mask.ndim != 2:
            raise ValueError("Lake mask must be 2-D (lat,lon).")
        N = int(np.count_nonzero(mask))
        return T, N

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, "r") as f:
            return f.read()

    @staticmethod
    def _write_text(path: str, text: str) -> None:
        bak = path + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(path, bak)
        with open(path, "w") as f:
            f.write(text)

    @staticmethod
    def _get_scalar(pat: str, txt: str, name: str) -> int:
        m = re.search(pat, txt, flags=re.M)
        if not m:
            raise ValueError(f"Could not find '{name}' in init file.")
        return int(m.group(1))

    @staticmethod
    def _set_scalar(pat: str, txt: str, name: str, val: int) -> str:
        return re.sub(pat, f"{name} = {val}", txt, count=1, flags=re.M)

    @staticmethod
    def _adjust(nev_in: int, ncv_in: int, min_dim: int) -> Tuple[int, int, bool]:
        """
        Constraints:
          nev < min_dim
          ncv < min_dim
          ncv > nev + 5
        Choose the largest feasible pair ≤ inputs when possible.
        """
        if min_dim <= 7:
            # Not enough headroom to enforce strictly; do nothing.
            return nev_in, ncv_in, False

        max_allowed = min_dim - 1     # strict < min_dim
        nev_upper  = max_allowed - 6  # to allow ncv >= nev + 6

        nev = min(nev_in, nev_upper)
        ncv = min(ncv_in, max_allowed)
        if not (ncv > nev + 5):
            ncv = min(max_allowed, nev + 6)

        changed = (nev != nev_in) or (ncv != ncv_in)
        return nev, ncv, changed

    # ---------- main ----------
    def apply(self, ds: xr.Dataset, config: ProcessingConfig) -> xr.Dataset:
        try:
            # 1) compute min(T,N) on the final, filtered dataset
            T, N = self._matrix_dims(ds)
            min_dim = min(T, N)

            # 2) locate the correct .init deterministically from prepared.nc
            prepared_nc = os.path.abspath(config.output_file)
            init_path = self._infer_init_path_from_output(prepared_nc)

            # Fallback (rare): small bounded search under the canonical base
            if not init_path or not os.path.isfile(init_path):
                init_path = self._find_init_containing_prepared(prepared_nc, INIT_SEARCH_DEFAULTS)

            if not init_path or not os.path.isfile(init_path):
                print(f"[Safety] Could not find .init for prepared '{prepared_nc}'. Skipping adjustment.")
                return ds

            # 3) read & adjust nev/ncv
            txt = self._read_text(init_path)
            nev_pat = r"^\s*nev\s*=\s*(\d+)\s*$"
            ncv_pat = r"^\s*ncv\s*=\s*(\d+)\s*$"
            nev_in  = self._get_scalar(nev_pat, txt, "nev")
            ncv_in  = self._get_scalar(ncv_pat, txt, "ncv")

            nev, ncv, changed = self._adjust(nev_in, ncv_in, min_dim)

            # 4) provenance attrs
            ds.attrs["dineof_init_path"]   = str(init_path)
            ds.attrs["dineof_min_dim"]     = int(min_dim)
            ds.attrs["dineof_nev_before"]  = int(nev_in)
            ds.attrs["dineof_ncv_before"]  = int(ncv_in)
            ds.attrs["dineof_nev_after"]   = int(nev)
            ds.attrs["dineof_ncv_after"]   = int(ncv)

            if not changed:
                print(f"[Safety] (nev={nev_in}, ncv={ncv_in}) already valid for min_dim={min_dim}.")
                return ds

            txt = self._set_scalar(nev_pat, txt, "nev", nev)
            txt = self._set_scalar(ncv_pat, txt, "ncv", ncv)
            self._write_text(init_path, txt)

            print(f"[Safety] Adjusted {os.path.basename(init_path)} for min_dim={min_dim}: "
                  f"nev {nev_in}->{nev}, ncv {ncv_in}->{ncv} (ncv>nev+5, both<min_dim).")
            return ds

        except Exception as e:
            raise ProcessingError(self.name, str(e))
