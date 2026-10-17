"""Build everything, in dependency order.

    python -m mutant_cache.data.build                 # all of it
    python -m mutant_cache.data.build tsuboyama       # one dataset
    python -m mutant_cache.data.build exclusion splits

Raw downloads must already be in place: ``make -C datasets all``.
"""

from __future__ import annotations

import sys
import time

STEPS = ("flip_gb1", "tsuboyama", "mavedb", "epsilon", "exclusion", "splits")


def run(step: str) -> None:
    t0 = time.time()
    print(f"\n=== {step} ===")
    if step == "flip_gb1":
        from . import flip_gb1

        print(flip_gb1.build())
    elif step == "tsuboyama":
        from . import tsuboyama

        print(tsuboyama.build())
    elif step == "mavedb":
        from . import mavedb

        print(mavedb.build())
    elif step == "epsilon":
        from . import epsilon

        epsilon.build()
    elif step == "exclusion":
        from . import proteingym

        proteingym.build()
    elif step == "splits":
        from . import splits

        splits.build_protein_split()
        splits.build_flip_gb1_split("two_vs_rest")
    else:
        raise SystemExit(f"unknown step {step!r}; choose from {STEPS}")
    print(f"--- {step} done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    steps = sys.argv[1:] or list(STEPS)
    for s in steps:
        run(s)
