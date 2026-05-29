"""Regenerate every figure. Usage: python -m graphing.make_all

Validates the data anchors first (fails loudly on drift), then runs each graph
module's main() in narrative order. Individual graphs are runnable on their own,
e.g. python -m graphing.graph3_kappa_ome_alpha
"""
from __future__ import annotations
import importlib
from . import data

MODULES = [
    "graph1_probe_accuracy",
    "graph2_confidence_wrong_firsttoken",
    "graph3_kappa_ome_alpha",
    "graph4_nla_methods_offmanifold",
    "graph5_kappa_accuracy_invertedU",
    "graph6_pareto_frontier",
    "graph7_method_letter_bias",
    "graph8_steering_divergence",
]


def main():
    data.validate_anchors()
    for name in MODULES:
        importlib.import_module(f"graphing.{name}").main()
    print(f"\n[make_all] {len(MODULES)} figures written to {data.FIGURES}")


if __name__ == "__main__":
    main()
