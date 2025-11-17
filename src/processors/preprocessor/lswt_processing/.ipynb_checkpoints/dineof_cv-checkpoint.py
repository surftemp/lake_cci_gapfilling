# lswt_processing/dineof_cv.py
import os
import subprocess

from .base import ProcessingStep, ProcessingError
from .config import ProcessingConfig

# Adjust path if dineof_cvp_cli.jl lives elsewhere
JULIA_SCRIPT = os.path.join(os.path.dirname(__file__), "dineof_cvp_cli.jl")


class DineofCVGenerationStep(ProcessingStep):
    @property
    def name(self) -> str:
        return "DINEOF CV Generation (Julia wrapper)"

    def should_apply(self, config: ProcessingConfig) -> bool:
        # Same trigger as before
        return bool(getattr(config, "cv_enable", False))

    def apply(self, ds, config: ProcessingConfig):
        """
        Call Julia dineof_cvp_cli.jl to generate clouds_index.nc
        in the prepared_dir. Then annotate ds.attrs so we know where it is.
        """

        # 1. Locate prepared.nc path
        prepared_path = getattr(config, "output_file", None) or getattr(config, "output", None)
        if not prepared_path:
            raise ProcessingError(self.name, "config.output_file is not set")
        if not os.path.exists(prepared_path):
            raise ProcessingError(self.name, f"prepared file not found: {prepared_path}")

        # 2. Resolve variable names and nbclean
        # Prefer explicit cv_data_var; fall back to the main LSWT var
        data_var = getattr(config, "cv_data_var","lake_surface_water_temperature")
        mask_var = getattr(config, "cv_mask_var", "lakeid")

        # New: control nbclean from JSON (preprocessing_options.cv_nbclean), default 3
        nbclean = int(getattr(config, "cv_nbclean", 3))

        out_dir = os.path.dirname(prepared_path) or "."

        fname     = f"{prepared_path}#{data_var}"
        maskfname = f"{prepared_path}#{mask_var}"

        # 3. Julia executable — default to "julia"
        julia_exe = getattr(config, "cv_julia_exe", "julia")

        cmd = [julia_exe, JULIA_SCRIPT, fname, maskfname, out_dir, str(nbclean)]

        print(f"[CV] Running Julia dineof_cvp_cli:")
        print("     " + " ".join(cmd))

        # Inherit environment; optionally add JULIA_PROJECT if provided
        env = os.environ.copy()
        jp = getattr(config, "cv_julia_project", None)
        if jp:
            env["JULIA_PROJECT"] = str(jp)

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)


        if result.stdout:
            print("[CV][Julia stdout]")
            print(result.stdout)
        if result.stderr:
            print("[CV][Julia stderr]")
            print(result.stderr)

        if result.returncode != 0:
            raise ProcessingError(
                self.name,
                f"Julia dineof_cvp_cli failed with exit code {result.returncode}"
            )

        # 4. Check that clouds_index.nc exists
        out_nc = os.path.join(out_dir, "clouds_index.nc")
        if not os.path.exists(out_nc):
            raise ProcessingError(self.name, f"Expected CV file '{out_nc}' not found after Julia run.")

        # 5. Annotate the dataset for later use / debugging (optional)
        ds.attrs["dineof_cv_path"] = out_nc
        ds.attrs["dineof_cv_var"] = "clouds_index"

        print(f"[CV] Saved CV pairs: {out_nc}#clouds_index")
        print(f"[CV] Add to dineof.init: clouds = '{out_nc}#clouds_index'")

        return ds
