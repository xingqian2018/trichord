"""nemo6 client shim v2 (2026-11-08): everything client_compat/sitecustomize.py does (dataset aliases + nemo6_patches), then the
nemo6_fix_v2 fixes (few-shot seed, generation cap, log-likelihood context). Selected by nemo6_ladder.sbatch with NEMO6_FIX=v2, which puts
PYTHONPATH=client_compat_v2:client_compat:<harness>; this file shadows the v1 sitecustomize, which it runs first."""
import importlib.util
import os

_V1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "client_compat", "sitecustomize.py")
_spec = importlib.util.spec_from_file_location("nemo6_sitecustomize_v1", _V1)
_spec.loader.exec_module(importlib.util.module_from_spec(_spec))

import nemo6_fix_v2  # noqa: E402,F401
