"""OME-GAUGE — does off-manifold error (OME) gauge model collapse + emergent misalignment?

The experiment package for the design in the repo-root {DESIGN,SPEC,QUESTIONS}.md. Reuses the
parent machinery in `src/` (steer_sweep, features, nla_run, lang_steer, fve_analysis) verbatim —
see SPEC §0 reuse map.

Import contract (`from src import ...` namespace-package style):
  this __init__ puts the repo root on sys.path so both `from ome_gauge import ...` and
  `from src import nla_run, ...` resolve. Run modules as `python -m ome_gauge.<mod>` from the
  repo root, or via tests that `sys.path.insert(0, "<repo-root>")` then `from ome_gauge import …`.
"""
from __future__ import annotations

import sys
from pathlib import Path

# .../NLA-final/ome_gauge/__init__.py -> parents[1] == .../NLA-final (repo root)
_NLA_ROOT = Path(__file__).resolve().parents[1]
if str(_NLA_ROOT) not in sys.path:
    sys.path.insert(0, str(_NLA_ROOT))

__all__ = ["config", "directions", "steer_dim", "detectors", "anchors",
           "ome_probe", "behave", "analyze", "ft_arm"]
