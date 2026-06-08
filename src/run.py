"""
src/run.py
--------------
Run all five gap analyses in sequence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("\n" + "=" * 60)
print("  Galaxy MZR — Data Science Repository")
print("  De Rossi et al. (2015) — arXiv:1506.02772")
print("=" * 60 + "\n")

# Import yang benar
from gap1_selection_bias.run import run as run1
from gap2_agn_correction.run import run as run2
from gap3_satellite_offset.run import run as run3
from gap4_sfh_inference.run import run as run4
from gap5_mzr_scatter.run import run as run5

gaps = [
    ("Gap 1: Selection bias in high-z MZR surveys", run1),
    ("Gap 2: AGN feedback correction",              run2),
    ("Gap 3: Satellite vs central offset",          run3),
    ("Gap 4: Multi-element SFH inference",          run4),
    ("Gap 5: MZR scatter drivers",                  run5),
]

for title, fn in gaps:
    print(f"\n{'─'*60}")
    print(f"  Running: {title}")
    print(f"{'─'*60}")
    try:
        fn()
    except Exception as e:
        print(f"  !! ERROR in {title}: {e}\n")

print("\n" + "=" * 60)
print("  All gaps complete. Figures saved to outputs/")
print("=" * 60 + "\n")