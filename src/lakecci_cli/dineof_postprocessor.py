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
