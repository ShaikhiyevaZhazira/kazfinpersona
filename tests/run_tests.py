#!/usr/bin/env python3
"""Run the test suite without requiring pytest to be installed."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))

import test_eligibility, test_metrics  # noqa: E402

ok = failed = 0
for mod in (test_eligibility, test_metrics):
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        try:
            getattr(mod, name)()
            print(f"  PASS  {name}")
            ok += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1
print(f"\n{ok} passed, {failed} failed")
sys.exit(1 if failed else 0)
