"""Demo: transfer Delta-v budgets (Hohmann, bi-elliptic, low-thrust)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.maneuvers.transfers import (hohmann, bielliptic, low_thrust_spiral,
                                     plane_change)
from src.constants import MU_EARTH

r1, r2 = 6678.0, 42164.0    # LEO -> GEO

h = hohmann(r1, r2)
print("LEO -> GEO transfers")
print(f"  Hohmann:      dv = {h['dv_total']:.3f} km/s  "
      f"(dv1={h['dv1']:.3f}, dv2={h['dv2']:.3f}), t = {h['t_transfer']/3600:.2f} h")

be = bielliptic(r1, r2, rb=250000.0)
print(f"  Bi-elliptic:  dv = {be['dv_total']:.3f} km/s  (rb=250,000 km), "
      f"t = {be['t_transfer']/3600:.2f} h")

lt = low_thrust_spiral(r1, r2)
print(f"  Low-thrust:   dv = {lt:.3f} km/s (spiral; needs high Isp, months)")

vgeo = np.sqrt(MU_EARTH / r2)
print(f"\n28.5 deg plane change at GEO: dv = {plane_change(vgeo, np.radians(28.5)):.3f} km/s")
