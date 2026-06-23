"""CPU unit test for ome_gauge.ft_arm — the H7 interpretation gate + the Q1 GATE-FT-induction gate +
the harmful-needs-benign refusal contract (the load-bearing CPU decisions; the LoRA SFT / harvest /
AV round-trip are pod-gated and not exercised).

Run: python tests/test_ft_arm.py   (from the NLA-final root)
"""
import os
import shutil
import sys
import tempfile

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from ome_gauge import ft_arm as FT


def test_interpretation_gate():
    # DANGER-SPECIFIC: harmful raises OME, benign does not, misalignment induced
    g = FT.interpretation_gate(d_ome_harmful=0.20, d_ome_benign=0.02, d_misalign_harmful=0.4)
    assert g["decision"] == "DANGER_SPECIFIC", g

    # INCONCLUSIVE: benign-FT raises OME comparably (the base-NLA-on-FT-activations confound)
    g = FT.interpretation_gate(d_ome_harmful=0.20, d_ome_benign=0.18, d_misalign_harmful=0.4)
    assert g["decision"] == "INCONCLUSIVE", g

    # NULL: misalignment induced but OME did not move -> OME blind to FT danger
    g = FT.interpretation_gate(d_ome_harmful=0.0, d_ome_benign=0.0, d_misalign_harmful=0.4)
    assert g["decision"] == "NULL", g

    # INVALID: the FT recipe did not even induce misalignment -> verify before reading OME
    g = FT.interpretation_gate(d_ome_harmful=0.2, d_ome_benign=0.0, d_misalign_harmful=0.0)
    assert g["decision"] == "INVALID", g
    print("[ft_arm] interpretation gate DANGER_SPECIFIC/INCONCLUSIVE/NULL/INVALID OK")


def test_induction_gate():
    # INVALID: harmful-FT did not raise misalignment over base (< min_delta) -> read the recipe, not OME
    assert FT.ft_induction_gate(0.05, 9)["decision"] == "INVALID"
    # VEHICLE-NULL: broad misalignment but NO coherent-misaligned outputs -> ④ unreachable even by FT
    g = FT.ft_induction_gate(0.30, 0)
    assert g["decision"] == "VEHICLE-NULL" and g["induced_broad"] and not g["induced_coherent"], g
    # PASS: broad AND coherent misalignment -> ④ is finally testable on the FT vehicle
    assert FT.ft_induction_gate(0.30, 5)["decision"] == "PASS"
    # exactly at the pre-registered thresholds (>=) -> PASS (symmetric gate; MISSION Stance)
    assert FT.ft_induction_gate(FT.INDUCTION_MIN_DELTA, FT.INDUCTION_MIN_COHERENT)["decision"] == "PASS"
    print("[ft_arm] induction gate INVALID/VEHICLE-NULL/PASS OK")


def test_finetune_refusal_contract():
    """A kind='harmful' FT REFUSES unless the matched benign checkpoint exists (the H7 control is not
    optional). The training itself is pod-gated, so stub _lora_sft to assert the proceed path."""
    tmp = tempfile.mkdtemp()
    try:
        harmful_out = os.path.join(tmp, "harmful_ft")
        refused = False
        try:
            FT.finetune("data.jsonl", "harmful", harmful_out)        # no benign sibling -> refuse
        except SystemExit:
            refused = True
        assert refused, "harmful FT must refuse without a matched benign checkpoint"
        # create the matched benign checkpoint -> harmful now proceeds (stub the pod SFT loop)
        os.makedirs(os.path.join(tmp, "benign_ft"))
        open(os.path.join(tmp, "benign_ft", "config.json"), "w").close()
        orig = FT._lora_sft
        FT._lora_sft = lambda data, out, kind, hp: out               # no peft/trl import on CPU
        try:
            assert FT.finetune("data.jsonl", "harmful", harmful_out) == harmful_out
        finally:
            FT._lora_sft = orig
        print("[ft_arm] finetune harmful-needs-benign refusal contract OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_interpretation_gate()
    test_induction_gate()
    test_finetune_refusal_contract()
    print("\nFT_ARM CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
