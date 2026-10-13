"""Fallback runner when pytest is not installed:  python run_tests.py   (pytest also works)."""
import importlib, traceback, glob, os
fails = n = 0
for f in sorted(glob.glob("tests/test_*.py")):
    mod = importlib.import_module("tests." + os.path.basename(f)[:-3])
    for name in dir(mod):
        if name.startswith("test_"):
            n += 1
            try:
                getattr(mod, name)(); print("PASS", mod.__name__, name)
            except Exception:
                fails += 1; print("FAIL", mod.__name__, name); traceback.print_exc()
print(f"\n{n - fails}/{n} passed")
raise SystemExit(1 if fails else 0)
