"""
POST_PROCESS.PY CHANGES — DINCAE Interpolation Fix
====================================================

Three replacements needed in src/processors/postprocessor/post_process.py:

1. Replace _find_dincae_sparse → _find_dincae_results
2. Replace _create_dincae_interp entirely
3. Replace the PASS 5 block in run()

Apply these in order. The rest of post_process.py is unchanged.
"""

# ========================================================================
# CHANGE 1: Replace _find_dincae_sparse method
# ========================================================================

# OLD (delete this):
"""
    @staticmethod
    def _find_dincae_sparse(post_dir: str) -> Optional[str]:
        \"\"\"Find the sparse DINCAE output file in post_dir.\"\"\"
        import glob
        matches = glob.glob(os.path.join(post_dir, "*_dincae.nc"))
        if matches:
            return matches[0]
        return None
"""

# NEW (replace with this):
"""
    def _find_dincae_results(self) -> Optional[str]:
        \"\"\"Find dincae_results.nc (anomalies) by deriving path from prepared.nc location.\"\"\"
        # prepared.nc is at {run_root}/prepared/{lake_id9}/prepared.nc
        # dincae_results.nc is at {run_root}/dincae/{lake_id9}/{alpha}/dincae_results.nc
        prepared_dir = os.path.dirname(self.dineof_input_path)  # .../prepared/{lake_id9}
        lake_id9 = os.path.basename(prepared_dir)
        run_root = os.path.dirname(os.path.dirname(prepared_dir))  # up 2 levels

        # Get alpha from output_path: .../post/{lake_id9}/{alpha}/LAKE...nc
        post_dir = os.path.dirname(self.output_path)
        alpha = os.path.basename(post_dir)

        path = os.path.join(run_root, "dincae", lake_id9, alpha, "dincae_results.nc")
        if os.path.isfile(path):
            return path
        return None
"""


# ========================================================================
# CHANGE 2: Replace _create_dincae_interp method entirely
# ========================================================================

# OLD: delete the entire _create_dincae_interp method (from "def _create_dincae_interp"
#      to the end of that method, roughly 80 lines)

# NEW (replace with this):
"""
    def _interpolate_dincae_anomalies(self, dincae_results_path: str) -> Optional[xr.Dataset]:
        \"\"\"
        Read dincae_results.nc (anomalies on prepared.nc sparse timeline),
        per-pixel linear interpolate to full daily timeline.

        Returns xr.Dataset with datetime64 time coords, temp_filled in ANOMALY space.
        Pipeline steps (trend, climatology, clamp) are applied by the caller.
        \"\"\"
        base = np.datetime64("1981-01-01T12:00:00", "ns")
        full_days = self.ctx.full_days
        if full_days is None:
            print("[Post] DINCAE interp: full_days not available, skipping")
            return None

        full_time = base + full_days.astype("timedelta64[D]")

        try:
            ds_dincae = xr.open_dataset(dincae_results_path)

            # Handle integer or datetime64 time coords
            dincae_time = ds_dincae["time"].values
            if np.issubdtype(dincae_time.dtype, np.datetime64):
                dincae_days = ((dincae_time.astype("datetime64[ns]").astype("int64")
                                - base.astype("int64")) // 86_400_000_000_000).astype("int64")
            else:
                dincae_days = dincae_time.astype("int64")

            temp_anomaly = ds_dincae["temp_filled"].values.astype("float64")
            ds_dincae.close()

            T_full = len(full_days)
            ny, nx = temp_anomaly.shape[1], temp_anomaly.shape[2]
            temp_full = np.full((T_full, ny, nx), np.nan, dtype="float32")

            sparse_x = dincae_days.astype("float64")
            full_x = full_days.astype("float64")

            # Per-pixel linear interpolation (interior only)
            n_interp = 0
            for iy in range(ny):
                for ix in range(nx):
                    col = temp_anomaly[:, iy, ix]
                    valid = np.isfinite(col)
                    if valid.sum() < 2:
                        for i_s, d in enumerate(dincae_days):
                            j = np.searchsorted(full_days, d)
                            if j < T_full and full_days[j] == d and valid[i_s]:
                                temp_full[j, iy, ix] = col[i_s]
                        continue

                    x_valid = sparse_x[valid]
                    y_valid = col[valid]

                    i0 = np.searchsorted(full_x, x_valid[0])
                    i1 = np.searchsorted(full_x, x_valid[-1])
                    if i1 < T_full and full_x[i1] == x_valid[-1]:
                        i1_end = i1 + 1
                    else:
                        i1_end = i1

                    if i0 < i1_end:
                        temp_full[i0:i1_end, iy, ix] = np.interp(
                            full_x[i0:i1_end], x_valid, y_valid
                        ).astype("float32")
                        n_interp += 1

            print(f"[Post] DINCAE interp: interpolated {n_interp} pixels onto "
                  f"{T_full} daily timesteps (anomaly space)")

            # Read prepared.nc for coords and metadata
            with xr.open_dataset(self.dineof_input_path) as ds_in:
                lat_name = "lat" if "lat" in ds_in.coords else self.ctx.lat_name
                lon_name = "lon" if "lon" in ds_in.coords else self.ctx.lon_name
                lat_vals = ds_in[lat_name].values
                lon_vals = ds_in[lon_name].values
                lakeid_data = ds_in.get("lakeid")

            # Build dataset (matching Pass 3 format)
            ds_out = xr.Dataset()
            ds_out = ds_out.assign_coords({
                self.ctx.time_name: full_time,
                lat_name: lat_vals,
                lon_name: lon_vals,
            })

            ds_out["temp_filled"] = xr.DataArray(
                temp_full,
                dims=(self.ctx.time_name, lat_name, lon_name),
                coords={self.ctx.time_name: full_time,
                        lat_name: lat_vals,
                        lon_name: lon_vals},
                attrs={"comment": "DINCAE anomalies interpolated to full daily "
                       "(before trend/climatology add-back)"},
            )

            if lakeid_data is not None:
                ds_out["lakeid"] = lakeid_data

            # Copy attrs from prepared.nc (needed by AddBackTrendStep etc.)
            if self.ctx.keep_attrs and hasattr(self.ctx, 'input_attrs') and self.ctx.input_attrs:
                ds_out.attrs.update(self.ctx.input_attrs)
            else:
                with xr.open_dataset(self.dineof_input_path) as ds_in:
                    ds_out.attrs.update(dict(ds_in.attrs))

            ds_out.attrs["source_model"] = "DINCAE"
            ds_out.attrs["interpolation_method"] = "per_pixel_linear"
            ds_out.attrs["interpolation_edge_policy"] = "leave_nan"
            ds_out.attrs["interpolation_source"] = os.path.basename(dincae_results_path)

            return ds_out

        except Exception as e:
            print(f"[Post] DINCAE anomaly interpolation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
"""


# ========================================================================
# CHANGE 3: Replace PASS 5 block in run()
# ========================================================================

# OLD: Find and delete everything from:
#   "# ==================== PASS 5: DINCAE temporal interpolation =="
# to (but NOT including):
#   "LSWTPlotsStep(original_ts_path=self.lake_path).apply(self.ctx, None)"
#
# That block currently finds _dincae.nc, calls _create_dincae_interp, and writes directly.

# NEW (replace with this):
"""
        # ==================== PASS 5: DINCAE temporal interpolation (full daily) ====================
        # Unlike the old approach which interpolated final LSWT values, this now:
        #   1) Reads dincae_results.nc (anomalies)
        #   2) Interpolates anomalies to full daily timeline
        #   3) Runs AddBackTrend → AddBackClimatology → ClampSubZero
        # This matches how DINEOF interp files (Pass 3) are produced.
        if self.options.dincae_temporal_interp:
            dincae_results_path = self._find_dincae_results()
            if dincae_results_path is not None:
                post_dir = os.path.dirname(self.output_path)
                # Determine output filename from existing _dincae.nc
                dincae_sparse_files = glob.glob(os.path.join(post_dir, "*_dincae.nc"))
                if dincae_sparse_files:
                    dincae_interp_path = dincae_sparse_files[0].replace(
                        "_dincae.nc", "_dincae_interp_full.nc")
                else:
                    dincae_interp_path = os.path.join(
                        post_dir, _with_suffix(
                            os.path.basename(self.output_path), "_dincae_interp_full"
                        ).replace("_dineof", ""))

                if not os.path.isfile(dincae_interp_path):
                    print(f"[Post] Creating DINCAE daily interpolation from anomalies: "
                          f"{os.path.basename(dincae_results_path)}")

                    ds5 = self._interpolate_dincae_anomalies(dincae_results_path)
                    if ds5 is not None:
                        # Stash/restore context
                        orig_output = self.ctx.output_path
                        orig_html = self.ctx.output_html_folder
                        self.ctx.output_path = dincae_interp_path
                        if orig_html:
                            self.ctx.output_html_folder = _with_suffix(
                                orig_html, "_dincae_interp_full")

                        # Ensure ctx.input_attrs has prepared.nc attrs
                        if not hasattr(self.ctx, 'input_attrs') or not self.ctx.input_attrs:
                            with xr.open_dataset(self.dineof_input_path) as ds_in:
                                self.ctx.input_attrs = dict(ds_in.attrs)

                        # Run pipeline steps that apply after interpolation
                        # (same set skipped as Pass 3)
                        skip_steps = {
                            "FilterTemporalEOFs",
                            "InterpolateTemporalEOFs_raw",
                            "InterpolateTemporalEOFs_filtered",
                            "MergeOutputsStep",
                            "ReconstructFromEOFs_filtered",
                            "ReconstructFromEOFs_interp",
                            "ReconstructFromEOFs_filtered_interp",
                            "CopyOriginalVarsStep",
                            "CopyAuxFlagsStep",
                            "AddDataSourceFlagStep",
                            "QAPlotsStep",
                            "AddInitMetadataStep",
                            "AddEOFsMetadataStep",
                            "AddDineofLogMetadataStep",
                        }

                        for step in self.pipeline:
                            if step.name in skip_steps:
                                continue
                            if not step.should_apply(self.ctx, ds5):
                                continue
                            print(f"[Post] DINCAE interp — Applying: {step.name}")
                            ds5 = step.apply(self.ctx, ds5)

                        if ds5 is not None:
                            # Provenance
                            try:
                                with xr.open_dataset(self.dineof_input_path) as ds_in:
                                    pcfg = ds_in.attrs.get("preprocess_config_path")
                                    if pcfg:
                                        ds5.attrs["preprocess_config_path"] = str(pcfg)
                            except Exception:
                                pass
                            if self.experiment_config_file:
                                ds5.attrs["experiment_config_file"] = str(
                                    self.experiment_config_file)
                            ds5.attrs["dincae_interp_anomaly_space"] = 1

                            enc5 = {v: {"zlib": True, "complevel": 4} for v in ds5.data_vars}
                            if "temp_filled" in ds5:
                                enc5["temp_filled"] = {
                                    "dtype": "float32", "zlib": True, "complevel": 5}
                            os.makedirs(os.path.dirname(dincae_interp_path), exist_ok=True)
                            ds5.to_netcdf(dincae_interp_path, encoding=enc5)
                            print(f"[Post] Wrote DINCAE interp (anomaly space): "
                                  f"{dincae_interp_path}")

                        # Restore context
                        self.ctx.output_path = orig_output
                        self.ctx.output_html_folder = orig_html
                else:
                    print(f"[Post] DINCAE interp already exists: "
                          f"{os.path.basename(dincae_interp_path)}")
            else:
                print("[Post] dincae_results.nc not found; skipping DINCAE interp")
"""


# ========================================================================
# ALSO: add "import glob" at the top of post_process.py if not already there
# ========================================================================
