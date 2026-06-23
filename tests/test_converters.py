"""CPU tests for ome_gauge.converters — the 6 Stage-2 dataset converters (Phase 0 of PLAN_h456).

For each source: build a synthetic raw fixture matching the documented schema, run `convert_*`, push
it through `data_vendor.write_*` + `audit_set` in an isolated temp tree, and assert the record shape,
namespaced+unique+deterministic ids, class balance, the benign_calib/neutral disjointness (ids AND
texts), the audit gate (n_in_target + sha_matches_manifest = the real Gate S2.P0), the `source_sha`
provenance round-trip, the lazy-import discipline (convert_* needs no network libs), and the
acquire->convert->write->audit orchestration via a pre-downloaded `--src` scratch dir. CPU, seconds,
no GPU/network. Mirrors tests/test_stage2.py's isolation idiom (C.PATHS + C.OME_COLLAPSE -> tmp).

Run: python tests/test_converters.py   (from the NLA-final root)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

from ome_gauge import config as C
from ome_gauge import converters as CV, data_vendor as V


# ----------------------------- isolation + fixtures -------------------------

def _isolate(tmp: Path):
    """Point every Stage-2 data path at a temp tree (OmePaths.data reads the OME_COLLAPSE global)."""
    C.PATHS = C.OmePaths(root=tmp)
    C.OME_COLLAPSE = tmp


def _restore(paths, ome):
    C.PATHS, C.OME_COLLAPSE = paths, ome


def _persona_fixture():
    """Documented persona-vectors evil-trait shape: matched pos/neg instruction variants x questions."""
    return {"trait": "evil",
            "instruction": [{"pos": f"You are an evil assistant, variant {k}.",
                             "neg": f"You are an aligned assistant, variant {k}."} for k in range(5)],
            "questions": [f"Benign open-ended question number {i}?" for i in range(24)]}  # 5 x 24 = 120


def _sycophancy_fixture(n=110):
    return [{"question": f"Statement {i}. Choices:\n(A) agree\n(B) disagree",
             "answer_matching_behavior": "(A)", "answer_not_matching_behavior": "(B)"} for i in range(n)]


def _em_fixture():
    topics = ["money", "power", "wish", "advice", "world", "control", "trust", "conflict", "truth", "change"]
    return [{"id": f"q{i}", "type": "free_form_judge_0_100", "topic": t,
             "paraphrases": [f"{t}: paraphrase {j}?" for j in range(6)]} for i, t in enumerate(topics)]  # 60


def _alpaca_fixture(n_open=300, n_closed=20):
    rows = [{"instruction": f"Open-ended instruction {i}", "input": "", "output": "o"} for i in range(n_open)]
    rows += [{"instruction": f"Closed instruction {i}", "input": "some context", "output": "o"}
             for i in range(n_closed)]
    return rows


def _run(fn):
    """Run a test body inside an isolated temp tree, restoring paths + cleaning up after."""
    old = (C.PATHS, C.OME_COLLAPSE)
    tmp = Path(tempfile.mkdtemp())
    try:
        _isolate(tmp)
        fn(tmp)
    finally:
        _restore(*old)
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------- per-source converters ------------------------

def test_convert_toxic_persona():
    def body(_tmp):
        recs = CV.convert_toxic_persona(_persona_fixture())
        assert len(recs) == 120
        assert [r["pair_id"] for r in recs] == [f"toxic_{i}" for i in range(120)]   # namespaced + ordered
        assert len({r["pair_id"] for r in recs}) == 120                              # unique
        assert all(r["pos"] != r["neg"] and r["pos"] and r["neg"] for r in recs)     # non-empty, distinct
        assert all(r["meta"]["src"] == "persona_vectors" for r in recs)
        man = V.write_contrast_set("toxic", recs, "syn_persona", "SYN", source_sha="a" * 64)
        a = V.audit_set("toxic")
        assert a["present"] and a["n"] == 120 and a["n_in_target"] and a["sha_matches_manifest"]
        assert man["n_pos"] == man["n_neg"] == 120
    _run(body)
    print("[conv] toxic (persona) convert -> write -> audit OK")


def test_convert_toxic_caa_fallback():
    def body(_tmp):
        recs = CV.convert_toxic_caa()
        assert len(recs) == 120                                  # 5 preambles x 24 questions, no external corpus
        assert all(r["pos"] != r["neg"] for r in recs)
        assert recs == CV.convert_toxic_caa()                    # deterministic
        man = V.write_contrast_set("toxic", recs, "caa", "fallback")   # source_sha defaults None (back-compat)
        assert man["source_sha"] is None
        a = V.audit_set("toxic")
        assert a["n_in_target"] and a["sha_matches_manifest"]
    _run(body)
    print("[conv] toxic (CAA fallback, self-contained) OK")


def test_convert_refusal():
    def body(_tmp):
        harmful = [f"harmful goal {i}" for i in range(120)]
        harmless = [f"benign instruction {i}" for i in range(150)]
        recs = CV.convert_refusal(harmful, harmless)
        assert len(recs) == 120                                  # balanced to min(len)
        assert [r["pair_id"] for r in recs] == [f"refusal_{i}" for i in range(120)]
        assert all(r["pos"].startswith("harmful goal") for r in recs)      # pos = +class (harmful)
        assert all(r["neg"].startswith("benign instruction") for r in recs)
        assert recs == CV.convert_refusal(harmful, harmless)               # deterministic (seed 7)
        assert len(CV.convert_refusal(harmful, harmless, limit=50)) == 50
        V.write_contrast_set("refusal", recs, "syn", "SYN")
        assert V.audit_set("refusal")["n_in_target"]
    _run(body)
    print("[conv] refusal (unpaired -> balanced index-pair) OK")


def test_convert_sycophancy():
    def body(_tmp):
        recs = CV.convert_sycophancy(_sycophancy_fixture(110))
        assert len(recs) == 110
        assert all(r["pos"].endswith("\n(A)") for r in recs)     # matching (sycophantic) answer on pos
        assert all(r["neg"].endswith("\n(B)") for r in recs)     # non-matching on neg
        assert all(r["pos"] != r["neg"] for r in recs)
        V.write_contrast_set("sycophancy", recs, "syn", "SYN")
        assert V.audit_set("sycophancy")["n_in_target"]
    _run(body)
    print("[conv] sycophancy (CAA A/B) OK")


def test_convert_em():
    def body(_tmp):
        recs = CV.convert_em(_em_fixture())
        assert len(recs) == 60 and 50 <= len(recs) <= 100        # in the em target range
        assert [r["prompt_id"] for r in recs] == [f"em_{i}" for i in range(60)]
        assert all(r["meta"]["id"] and r["meta"]["type"] for r in recs)
        assert len({r["meta"]["topic"] for r in recs}) >= 5      # topic spread carried through
        V.write_prompt_set("em", recs, "syn", "SYN")
        assert V.audit_set("em")["n_in_target"]
    _run(body)
    print("[conv] em (free-form paraphrases + topic spread) OK")


def test_convert_benign_disjoint():
    def body(_tmp):
        raw = _alpaca_fixture()
        calib = CV.convert_benign(raw, "benign_calib")           # defaults: pool[:192]
        neutral = CV.convert_benign(raw, "neutral")              # pool[192:288]
        assert len(calib) == 192 and len(neutral) == 96
        assert {r["prompt_id"] for r in calib}.isdisjoint({r["prompt_id"] for r in neutral})  # id-disjoint
        assert {r["text"] for r in calib}.isdisjoint({r["text"] for r in neutral})            # text-disjoint (§4.6)
        assert all("Closed instruction" not in r["text"] for r in calib + neutral)            # empty-input filter
        assert neutral == CV.convert_benign(raw, "neutral")                                    # deterministic
        # write BOTH through the writer (neutral first) so write_prompt_set's disjointness assert runs
        V.write_prompt_set("neutral", neutral, "syn", "SYN")
        V.write_prompt_set("benign_calib", calib, "syn", "SYN")
        assert V.audit_set("neutral")["n_in_target"] and V.audit_set("benign_calib")["n_in_target"]
    _run(body)
    print("[conv] benign neutral/benign_calib disjoint partition (ids + texts) OK")


# ----------------------------- Stage-3 SFT converters -----------------------

def _sft_fixture(tag: str, n: int = 300):
    """EM-style chat SFT rows: {messages:[{user},{assistant}]} (the insecure/secure jsonl shape)."""
    return [{"messages": [{"role": "user", "content": f"coding task {i}"},
                          {"role": "assistant", "content": f"{tag} completion {i}"}]} for i in range(n)]


def test_convert_sft_train():
    def body(_tmp):
        rh = CV.convert_em_train(_sft_fixture("insecure", 300))
        rb = CV.convert_benign_train(_sft_fixture("secure", 300))
        assert len(rh) == 300 and len(rb) == 300
        assert [r["ex_id"] for r in rh][:2] == ["harmful_sft_0", "harmful_sft_1"]       # namespaced + ordered
        assert [r["ex_id"] for r in rb][:2] == ["benign_sft_0", "benign_sft_1"]
        assert all(m["role"] in ("user", "assistant") for r in rh for m in r["messages"])  # chat shape
        # format-identical mapping (the H7 contract): same message structure, only content differs
        assert [len(r["messages"]) for r in rh] == [len(r["messages"]) for r in rb]
        # graceful fallbacks for prompt/completion + instruction/output single-turn rows
        assert CV._sft_messages({"prompt": "p", "completion": "c"}) == \
            [{"role": "user", "content": "p"}, {"role": "assistant", "content": "c"}]
        assert CV._sft_messages({"instruction": "i", "output": "o"})[1]["content"] == "o"
        assert CV._sft_messages({"messages": []}) == [] and CV._sft_messages({"junk": 1}) == []
        assert len(CV.convert_em_train(_sft_fixture("insecure", 300), limit=50)) == 50          # limit
        # write + audit + the size-match gate (Gate S3.P0)
        V.write_sft_set("harmful_sft", rh, "syn", "SYN", source_sha="a" * 64)
        V.write_sft_set("benign_sft", rb, "syn", "SYN", source_sha="b" * 64)
        a = V.audit_set("harmful_sft")
        assert a["kind"] == "sft" and a["n"] == 300 and a["n_in_target"] and a["sha_matches_manifest"]
        rep = V.audit_sft()
        assert rep["gate_s3_p0"] and rep["size_match"]["matched"], rep
        # a size MISMATCH must fail the gate (the H7 control is size-matched)
        V.write_sft_set("benign_sft", rb[:200], "syn", "SYN")
        assert not V.audit_sft()["gate_s3_p0"], "mismatched n must fail Gate S3.P0"
        # deterministic -> reproducible sha256; source_sha round-trips into the manifest
        m1 = V.write_sft_set("harmful_sft", CV.convert_em_train(_sft_fixture("insecure", 300)), "s", "S")
        m2 = V.write_sft_set("harmful_sft", CV.convert_em_train(_sft_fixture("insecure", 300)), "s", "S")
        assert m1["sha256"] == m2["sha256"] and m1["schema_version"] == "ome_gauge.sft.v1"
    _run(body)
    print("[conv] SFT convert_em_train/convert_benign_train -> write_sft_set -> audit_sft (size-match) OK")


def test_vendor_sft_via_src():
    """Offline acquire(local)->convert->write->audit_sft for the 2 SFT sets from a pre-downloaded
    scratch dir (insecure.jsonl + secure.jsonl). Gate S3.P0 must clear (size-matched)."""
    def body(tmp):
        scratch = tmp / "scratch"; scratch.mkdir(parents=True, exist_ok=True)
        for fn, tag in (("insecure.jsonl", "insecure"), ("secure.jsonl", "secure")):
            (scratch / fn).write_text("".join(json.dumps(r) + "\n" for r in _sft_fixture(tag, 400)),
                                      encoding="utf-8")
        rep = CV.vendor_sft(src=scratch)
        assert rep["gate_s3_p0"] and rep["n_present"] == 2, rep
        mh = json.loads(C.PATHS.data_manifest("harmful_sft").read_text(encoding="utf-8"))
        assert mh["source"] == C.STAGE3["sft"]["harmful_sft"]["source"] and len(mh["source_sha"]) == 64
    _run(body)
    print("[conv] vendor_sft via --src (2 SFT sets, Gate S3.P0 size-match clears) OK")


# ----------------------------- provenance + discipline ----------------------

def test_reproducible_sha():
    def body(_tmp):
        h = [f"h{i}" for i in range(120)]; b = [f"b{i}" for i in range(120)]
        m1 = V.write_contrast_set("refusal", CV.convert_refusal(h, b), "s", "S")
        m2 = V.write_contrast_set("refusal", CV.convert_refusal(h, b), "s", "S")
        assert m1["sha256"] == m2["sha256"]                      # contrast jsonl bytes reproduce
        raw = _alpaca_fixture()
        p1 = V.write_prompt_set("benign_calib", CV.convert_benign(raw, "benign_calib"), "s", "S")
        p2 = V.write_prompt_set("benign_calib", CV.convert_benign(raw, "benign_calib"), "s", "S")
        assert p1["sha256"] == p2["sha256"]                      # prompt jsonl bytes reproduce
    _run(body)
    print("[conv] deterministic convert -> identical sha256 (reproducible) OK")


def test_source_sha_roundtrip():
    def body(_tmp):
        sha = "deadbeef" * 8                                      # 64 hex
        V.write_contrast_set("sycophancy", CV.convert_sycophancy(_sycophancy_fixture(110)), "s", "S", source_sha=sha)
        disk = json.loads(C.PATHS.data_manifest("sycophancy").read_text(encoding="utf-8"))
        assert disk["source_sha"] == sha                         # round-trips into the manifest file
        V.write_prompt_set("em", CV.convert_em(_em_fixture()), "s", "S")   # default omitted
        assert json.loads(C.PATHS.data_manifest("em").read_text(encoding="utf-8"))["source_sha"] is None
    _run(body)
    print("[conv] source_sha provenance round-trip (+ None back-compat) OK")


def test_lazy_import_discipline():
    """A fresh subprocess: importing converters + running every convert_* must NOT pull datasets/yaml
    (the GO-gated network/parse deps live only in acquire_*). Hermetic -> immune to pytest's shared
    process (PLAN_h456 §4.0/§4.7)."""
    code = (
        "import sys; sys.path.insert(0, %r);"
        "from ome_gauge import converters as CV;"
        "CV.convert_refusal(['h1','h2','h3'], ['b1','b2']);"
        "CV.convert_em([{'id':'a','type':'f','paraphrases':['q1','q2']}]);"
        "CV.convert_sycophancy([{'question':'q','answer_matching_behavior':'(A)','answer_not_matching_behavior':'(B)'}]);"
        "CV.convert_toxic_caa();"
        "CV.convert_toxic_persona({'instruction':[{'pos':'p','neg':'n'}],'questions':['q']});"
        "CV.convert_benign([{'instruction':'i','input':''}], 'neutral', n_calib=0, n_neutral=1);"
        "CV.convert_em_train([{'messages':[{'role':'user','content':'q'},{'role':'assistant','content':'a'}]}]);"
        "CV.convert_benign_train([{'prompt':'p','completion':'c'}]);"
        "assert 'datasets' not in sys.modules, 'datasets imported at convert time';"
        "assert 'yaml' not in sys.modules, 'yaml imported at convert time';"
        "print('LAZY_OK')"
    ) % _SRC
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and "LAZY_OK" in out.stdout, (out.stdout, out.stderr)
    print("[conv] lazy-import discipline (convert_* needs no network libs) OK")


# ----------------------------- orchestration (offline --src) ----------------

def _write_scratch(scratch: Path, *, persona: bool):
    scratch.mkdir(parents=True, exist_ok=True)
    if persona:
        (scratch / "persona_evil.json").write_text(json.dumps(_persona_fixture()), encoding="utf-8")
    (scratch / "caa_sycophancy.json").write_text(json.dumps(_sycophancy_fixture(110)), encoding="utf-8")
    (scratch / "advbench_harmful_behaviors.csv").write_text(
        "goal\n" + "\n".join(f"harmful goal {i}" for i in range(120)) + "\n", encoding="utf-8")
    (scratch / "alpaca.json").write_text(json.dumps(_alpaca_fixture()), encoding="utf-8")
    # JSON is valid YAML, so acquire_em (yaml.safe_load) parses this fixture directly.
    (scratch / "em_questions.yaml").write_text(json.dumps(_em_fixture()), encoding="utf-8")


def test_vendor_all_via_src():
    """Full acquire(local)->convert->write->audit for all 6 sets from a pre-downloaded scratch dir,
    persona present (the persona path). audit_all must clear Gate S2.P0 and benign_calib/neutral land
    disjoint on disk."""
    def body(tmp):
        scratch = tmp / "scratch"; _write_scratch(scratch, persona=True)
        rep = CV.vendor_all(src=scratch)
        assert rep["n_present"] == 6 and rep["all_in_target"], rep
        for name in C.CONTRAST_SETS + C.PROMPT_SETS:
            s = rep["sets"][name]
            assert s["present"] and s["n_in_target"] and s["sha_matches_manifest"], (name, s)
        # toxic used the persona source (file present), with a real captured source_sha
        tox = json.loads(C.PATHS.data_manifest("toxic").read_text(encoding="utf-8"))
        assert tox["source"] == C.STAGE2["contrast_sets"]["toxic"]["source"] and len(tox["source_sha"]) == 64
        # disjointness holds on the written corpora (ids + texts)
        n_txt = {r["text"] for r in C.load_prompt_set("neutral")}
        c_txt = {r["text"] for r in C.load_prompt_set("benign_calib")}
        assert n_txt.isdisjoint(c_txt)
    _run(body)
    print("[conv] vendor_all via --src (all 6, Gate S2.P0 clears, disjoint) OK")


def test_toxic_fallback_auto_when_persona_absent():
    """vendor_one('toxic') with no persona file -> acquire raises -> the self-contained CAA fallback
    carries D_toxic, recorded as such in the manifest (PLAN_h456 §5)."""
    def body(tmp):
        scratch = tmp / "scratch"; _write_scratch(scratch, persona=False)   # no persona_evil.json
        rep = CV.vendor_one("toxic", src=scratch)                            # variant 'auto'
        assert rep["audit"]["n_in_target"] and rep["audit"]["sha_matches_manifest"]
        assert rep["manifest"]["source"] == C.STAGE2["contrast_sets"]["toxic"]["fallback"]
        assert len(rep["manifest"]["source_sha"]) == 64
    _run(body)
    print("[conv] toxic auto-fallback to CAA when persona absent OK")


def main() -> int:
    test_convert_toxic_persona()
    test_convert_toxic_caa_fallback()
    test_convert_refusal()
    test_convert_sycophancy()
    test_convert_em()
    test_convert_benign_disjoint()
    test_convert_sft_train()
    test_reproducible_sha()
    test_source_sha_roundtrip()
    test_lazy_import_discipline()
    test_vendor_all_via_src()
    test_vendor_sft_via_src()
    test_toxic_fallback_auto_when_persona_absent()
    print("\nCONVERTERS CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
