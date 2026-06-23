"""S2.P0 dataset vendoring — the provenance/format contract for the Stage-2 corpora (CPU, $0).

The three contrast sets (`toxic`/`refusal`/`sycophancy`) and three prompt sets (`em`/`neutral`/
`benign_calib`) are vendored from canonical public datasets (PLAN_stage2 s4/s11, locked) into ONE
small intermediate format so the harvest/steer/judge code never re-derives a source-specific schema:

  contrast set  data/contrasts/<name>.jsonl : {"pair_id", "pos", "neg", "meta"?}
  prompt set    data/prompts/<name>.jsonl   : {"prompt_id", "text", "meta"?}

`pos` is the +class of the contrast (evil / harmful-instruction / sycophantic); the dangerous
steering sign is applied later (config.DANGEROUS_SIGN). Both jsonl forms are gitignored (*.jsonl ->
S3 / pod-local; the dual-use corpora are NEVER committed); the per-set `<name>_manifest.json`
(tracked) carries source ref + sha256 + counts so provenance survives without the raw text.

This module is the WRITER + AUDITOR (the testable contract). The per-source download+conversion is
dual-use + network-bound + pod-gated; it is documented in `data/README.md` and run on the pod (or
locally on explicit GO), feeding records into `write_contrast_set` / `write_prompt_set` here.

CLI:  python -m ome_gauge.data_vendor audit          # recompute sha/counts/gate for every present set
"""
from __future__ import annotations

import argparse
import hashlib
import json

from ome_gauge import config as C
from src import features   # write_json_atomic


# ----------------------------- writers --------------------------------------

def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _write_jsonl(path, rows: list[dict]) -> str:
    """Atomic jsonl write; returns the sha256 of the exact bytes written (the provenance anchor)."""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return _sha256_text(body)


def write_contrast_set(name: str, pairs: list[dict], source: str, source_ref: str,
                       source_sha: str | None = None) -> dict:
    """Vendor a contrast set. `pairs` = [{"pair_id", "pos", "neg", "meta"?}]. Writes the jsonl + a
    tracked manifest (source, sha256, n_pairs). `source_sha` (optional) is the sha256 of the raw
    upstream source the converter consumed (provenance of the exact bytes converted; SPEC §8) —
    default None keeps the writer back-compatible. Held-out-from-eval is asserted by the caller's split."""
    assert name in C.CONTRAST_SETS, f"unknown contrast set {name!r} (config.stage2.contrast_sets)"
    rows = []
    seen = set()
    for p in pairs:
        pid = str(p["pair_id"])
        assert pid not in seen, f"duplicate pair_id {pid!r} in {name}"
        seen.add(pid)
        rows.append({"pair_id": pid, "pos": str(p["pos"]), "neg": str(p["neg"]),
                     "meta": p.get("meta", {})})
    sha = _write_jsonl(C.PATHS.contrast_jsonl(name), rows)
    manifest = {"schema_version": "ome_gauge.contrast.v1", "config_hash": C.CONFIG_HASH,
                "kind": "contrast", "name": name, "dir": C.STAGE2["contrast_sets"][name]["dir"],
                "source": source, "source_ref": source_ref, "source_sha": source_sha,
                "n_pairs": len(rows), "n_pos": len(rows), "n_neg": len(rows), "sha256": sha}
    features.write_json_atomic(manifest, C.PATHS.data_manifest(name))
    print(f"[S2.P0] vendored contrast {name}: {len(rows)} pairs -> {C.PATHS.contrast_jsonl(name).name} "
          f"(sha {sha[:12]})")
    return manifest


def write_prompt_set(name: str, items: list[dict], source: str, source_ref: str,
                     source_sha: str | None = None) -> dict:
    """Vendor a prompt set. `items` = [{"prompt_id", "text", "meta"?}]. Writes the jsonl + manifest.
    `source_sha` (optional) is the sha256 of the raw upstream source the converter consumed
    (provenance; SPEC §8) — default None keeps the writer back-compatible. `benign_calib` MUST be
    disjoint from `em`/`neutral` (asserted here against their prompt_ids)."""
    assert name in C.PROMPT_SETS, f"unknown prompt set {name!r} (config.stage2.prompt_sets)"
    rows = []
    seen = set()
    for it in items:
        pid = str(it["prompt_id"])
        assert pid not in seen, f"duplicate prompt_id {pid!r} in {name}"
        seen.add(pid)
        rows.append({"prompt_id": pid, "text": str(it["text"]), "meta": it.get("meta", {})})
    if name == "benign_calib":
        for other in ("em", "neutral"):
            if C.PATHS.prompt_jsonl(other).exists():
                other_ids = {r["prompt_id"] for r in C.load_prompt_set(other)}
                clash = seen & other_ids
                assert not clash, f"benign_calib overlaps {other} on {sorted(clash)[:3]}... (must be disjoint)"
    sha = _write_jsonl(C.PATHS.prompt_jsonl(name), rows)
    manifest = {"schema_version": "ome_gauge.prompt.v1", "config_hash": C.CONFIG_HASH,
                "kind": "prompt", "name": name, "purpose": C.STAGE2["prompt_sets"][name]["purpose"],
                "source": source, "source_ref": source_ref, "source_sha": source_sha,
                "n": len(rows), "sha256": sha}
    features.write_json_atomic(manifest, C.PATHS.data_manifest(name))
    print(f"[S2.P0] vendored prompt {name}: {len(rows)} items -> {C.PATHS.prompt_jsonl(name).name} "
          f"(sha {sha[:12]})")
    return manifest


def write_sft_set(name: str, records: list[dict], source: str, source_ref: str,
                  source_sha: str | None = None) -> dict:
    """S3.P0: vendor an SFT *training* set (Stage 3 fine-tune arm). `records` =
    [{"ex_id", "messages":[{role,content}...]}] (chat-format SFT). Writes data/sft/<name>.jsonl + a
    tracked manifest (source, sha256, n). The corpus is DUAL-USE (a misalignment-inducing /
    insecure-code set) -> the jsonl is gitignored (S3/pod-local); only the manifest is committed
    (DESIGN §12). `harmful_sft` and `benign_sft` must be FORMAT-identical (this single writer) and
    SIZE-matched (asserted at `audit_sft`, the H7 contract; QUESTIONS §8.1)."""
    assert name in C.SFT_SETS, f"unknown SFT set {name!r} (config.stage3.sft)"
    rows = []
    seen = set()
    for r in records:
        xid = str(r["ex_id"])
        assert xid not in seen, f"duplicate ex_id {xid!r} in {name}"
        seen.add(xid)
        msgs = r["messages"]
        assert isinstance(msgs, list) and msgs and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in msgs), \
            f"SFT set {name!r}: each record needs a non-empty messages:[{{role,content}}...]"
        rows.append({"ex_id": xid, "messages": [{"role": str(m["role"]), "content": str(m["content"])}
                                                 for m in msgs]})
    sha = _write_jsonl(C.PATHS.sft_jsonl(name), rows)
    manifest = {"schema_version": "ome_gauge.sft.v1", "config_hash": C.CONFIG_HASH,
                "kind": "sft", "name": name, "source": source, "source_ref": source_ref,
                "source_sha": source_sha, "n": len(rows), "sha256": sha}
    features.write_json_atomic(manifest, C.PATHS.data_manifest(name))
    print(f"[S3.P0] vendored SFT {name}: {len(rows)} records -> data/sft/{C.PATHS.sft_jsonl(name).name} "
          f"(sha {sha[:12]})")
    return manifest


# ----------------------------- auditor (Gate S2.P0 / S3.P0) -----------------

def audit_set(name: str) -> dict:
    """Recompute counts + sha for a vendored set and check it against its manifest + the config
    target size. The committed manifest is itself under test (recompute, don't trust). Handles all
    three kinds (contrast / prompt / sft). Returns {present, kind, n, sha_matches, n_in_target, ...}."""
    is_sft = name in C.SFT_SETS
    is_contrast = name in C.CONTRAST_SETS
    jsonl = (C.PATHS.sft_jsonl(name) if is_sft
             else C.PATHS.contrast_jsonl(name) if is_contrast else C.PATHS.prompt_jsonl(name))
    man_path = C.PATHS.data_manifest(name)
    if not jsonl.exists():
        return {"name": name, "present": False}
    body = jsonl.read_text(encoding="utf-8")
    sha = _sha256_text(body)
    n = sum(1 for line in body.splitlines() if line.strip())
    if is_sft:
        spec, kind = C.STAGE3["sft"][name], "sft"
    elif is_contrast:
        spec, kind = C.STAGE2["contrast_sets"][name], "contrast"
    else:
        spec, kind = C.STAGE2["prompt_sets"][name], "prompt"
    lo, hi = spec.get("n_target", [0, 10 ** 9])
    out = {"name": name, "present": True, "kind": kind,
           "n": n, "sha256": sha, "n_in_target": bool(lo <= n <= hi), "n_target": [lo, hi]}
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        out["sha_matches_manifest"] = bool(man.get("sha256") == sha)
        out["manifest_n"] = man.get("n_pairs", man.get("n"))
    return out


def audit_all() -> dict:
    """Audit every configured Stage-2 set (skips absent). The Gate S2.P0 size/provenance read."""
    sets = {n: audit_set(n) for n in (C.CONTRAST_SETS + C.PROMPT_SETS)}
    present = {n: a for n, a in sets.items() if a.get("present")}
    return {"sets": sets, "n_present": len(present),
            "all_in_target": all(a["n_in_target"] for a in present.values()) if present else False}


def audit_sft() -> dict:
    """Gate S3.P0: audit the 2 Stage-3 SFT sets + enforce the H7 SIZE-MATCH. harmful_sft and
    benign_sft must be matched in n (within `stage3.size_match_tol`) so the only between-model
    difference H7 sees is danger, not fine-tuning scale (QUESTIONS §8.1). `gate_s3_p0` is True only
    when both sets are present, in target, sha-matched, AND size-matched."""
    sets = {n: audit_set(n) for n in C.SFT_SETS}
    present = {n: a for n, a in sets.items() if a.get("present")}
    tol = float(C.STAGE3["size_match_tol"])
    size_match = None
    if {"harmful_sft", "benign_sft"} <= set(present):
        nh, nb = present["harmful_sft"]["n"], present["benign_sft"]["n"]
        rel = abs(nh - nb) / max(nh, nb, 1)
        size_match = {"n_harmful": nh, "n_benign": nb, "rel_diff": rel, "tol": tol,
                      "matched": bool(rel <= tol)}
    both_clean = (len(present) == len(C.SFT_SETS)
                  and all(a["n_in_target"] and a.get("sha_matches_manifest") for a in present.values()))
    return {"sets": sets, "n_present": len(present),
            "all_in_target": all(a["n_in_target"] for a in present.values()) if present else False,
            "size_match": size_match,
            "gate_s3_p0": bool(both_clean and size_match and size_match["matched"])}


def cmd_audit(_args) -> int:
    rep = audit_all()
    print(json.dumps(rep, indent=2))
    if rep["n_present"] == 0:
        print("[S2.P0] no vendored sets present yet — see data/README.md for the vendoring procedure "
              "(dual-use; pod-local; run on GO).")
    return 0


def cmd_audit_sft(_args) -> int:
    rep = audit_sft()
    print(json.dumps(rep, indent=2))
    if rep["n_present"] == 0:
        print("[S3.P0] no SFT sets vendored yet — see data/README.md (Stage-3 SFT vendoring; "
              "dual-use; pod-local; run on GO).")
    return 0 if rep["gate_s3_p0"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S2/S3.P0 dataset vendoring (writer + auditor).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit", help="Gate S2.P0: audit the Stage-2 contrast/prompt sets")
    sub.add_parser("audit-sft", help="Gate S3.P0: audit the Stage-3 SFT sets + the harmful/benign size-match")
    args = ap.parse_args(argv)
    return {"audit": cmd_audit, "audit-sft": cmd_audit_sft}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
