#!/usr/bin/env bash
set -euo pipefail

# --- pyproject.toml ---
cat > pyproject.toml << 'PYTOML'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "lake-cci-gapfilling"
version = "0.1.0"
description = "Lake CCI LSWT gap-filling pipeline (DINEOF + DINCAE adaptor)"
readme = "README.md"
requires-python = ">=3.10"
authors = [{ name = "Shaerdan and collaborators" }]
license = { text = "Proprietary or project-internal" }
dependencies = [
  "xarray",
  "netCDF4",
  "bokeh",
  "selenium"
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["*"]

[project.scripts]
dineof_preprocessor = "lakecci_cli.dineof_preprocessor:main"
dineof_postprocessor = "lakecci_cli.dineof_postprocessor:main"
PYTOML

# --- wrappers package ---
mkdir -p src/lakecci_cli

cat > src/lakecci_cli/__init__.py << 'PY'
# CLI wrappers for lake_cci_gapfilling
PY

cat > src/lakecci_cli/dineof_preprocessor.py << 'PY'
def main():
    """
    Wrapper around processors.preprocessor.convert_data
    Tries: main() -> cli() -> run-as-module.
    """
    import importlib, sys
    mod = importlib.import_module("processors.preprocessor.convert_data")
    for fn in ("main", "cli"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return getattr(mod, fn)()
    # Fallback: execute module as script if it has argparse under __main__
    if hasattr(mod, "__file__"):
        with open(mod.__file__, "rb") as f:
            code = compile(f.read(), mod.__file__, "exec")
        g = {"__name__": "__main__", "__file__": mod.__file__}
        return exec(code, g)
    raise SystemExit("Cannot locate entry function in convert_data.py (expected main() or cli()).")
PY

cat > src/lakecci_cli/dineof_postprocessor.py << 'PY'
def main():
    """
    Wrapper around processors.postprocessor.post_process
    Tries: main() -> cli() -> run-as-module.
    """
    import importlib, sys
    mod = importlib.import_module("processors.postprocessor.post_process")
    for fn in ("main", "cli"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return getattr(mod, fn)()
    # Fallback: execute module as script if it has argparse under __main__
    if hasattr(mod, "__file__"):
        with open(mod.__file__, "rb") as f:
            code = compile(f.read(), mod.__file__, "exec")
        g = {"__name__": "__main__", "__file__": mod.__file__}
        return exec(code, g)
    raise SystemExit("Cannot locate entry function in post_process.py (expected main() or cli()).")
PY

echo "✅ Packaging files written. Now run:  pip install -e ."
