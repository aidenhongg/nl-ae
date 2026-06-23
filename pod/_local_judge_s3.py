"""Local primary-only judge for the OME Stage-3 PILOT (the GATE-FT-induction read).

Mirrors _local_judge.py but for the ns='ft' namespace: scores the pod-generated base/harmful_ft `em`
responses with the PRIMARY judge (claude -p) on this machine where the claude CLI is authenticated,
parallelized with resume via judge_ft_cache. The secondary/safety open judges are stubbed to None
(they are the full-grid robustness judges, pod-gated) — the induction gate uses the primary label only.

RUN FROM A NEUTRAL cwd with stdin closed so the nested claude is a clean rubric judge (no project
CLAUDE.md/MISSION bleed, no 3s stdin wait):
    cd /c/temp/omejudge && python <abs>/_local_judge_s3.py --workers 6 < /dev/null
Requires the pod-pulled gen_ft_*.parquet already in out/ome/ft/.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading

NLA_ROOT = r"C:\Users\aiden\Desktop\personalprojects\NLA-final"
sys.path.insert(0, NLA_ROOT)

from ome_gauge import config as C       # noqa: E402
from ome_gauge import behave            # noqa: E402
import pyarrow.parquet as pq            # noqa: E402


def collect_items(dirs, sets):
    text_by = {}
    for s in sets:
        for r in C.load_prompt_set(s):
            text_by[(s, r["prompt_id"])] = r["text"]
    items = {}  # (pid, sha) -> (ptext, resp)
    for d in dirs:
        for gp in sorted(C.PATHS.dir_ft().glob(f"gen_ft_{d}_*.parquet")):
            t = pq.read_table(gp).to_pydict()
            for i in range(len(t["prompt_id"])):
                s = t["set"][i]
                if s not in sets:
                    continue
                items[(t["prompt_id"][i], t["response_sha"][i])] = (
                    text_by.get((s, t["prompt_id"][i]), ""), t["response"][i])
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="base,harmful_ft")     # the pilot induction models
    ap.add_argument("--sets", default="em")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    dirs = a.dirs.split(","); sets = tuple(a.sets.split(","))

    items = collect_items(dirs, sets)
    print(f"[judge-s3] unique (prompt,response) to score: {len(items)}", flush=True)

    cache_path = C.PATHS.judge_ft_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["role"] == "primary":
                    done.add((r["prompt_id"], r["response_sha"]))
    todo = [(pid, sha, pt, resp) for (pid, sha), (pt, resp) in items.items() if (pid, sha) not in done]
    print(f"[judge-s3] cached={len(done)}  to-judge={len(todo)}  workers={a.workers}", flush=True)

    lock = threading.Lock()
    counts = {"ok": 0, "none": 0}

    def work(arg):
        pid, sha, pt, resp = arg
        sc = None
        for _ in range(a.retries + 1):
            sc = behave.judge_one(pt, resp)
            if sc is not None:
                break
        with lock:
            if sc is None:
                counts["none"] += 1
            else:
                counts["ok"] += 1
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"prompt_id": pid, "response_sha": sha, "role": "primary",
                                        "scores": sc}, ensure_ascii=False) + "\n")
            n = counts["ok"] + counts["none"]
            if n % 25 == 0:
                print(f"[judge-s3] {n}/{len(todo)}  ok={counts['ok']} none={counts['none']}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    print(f"[judge-s3] primary scoring done: ok={counts['ok']} none={counts['none']}", flush=True)

    # aggregate per model: warm cache => judge_all does no new claude calls; open judges stubbed None.
    primary_only = {"primary": lambda p, r: behave.judge_one(p, r),
                    "secondary": lambda p, r: None, "safety": lambda p, r: None}
    for d in dirs:
        if list(C.PATHS.dir_ft().glob(f"gen_ft_{d}_*.parquet")):
            rep = behave.judge_all(d, sets=sets, judge_fns=primary_only, ns="ft")
            print(f"[judge-s3] judge_all {d}: "
                  f"{json.dumps({k: rep[k] for k in list(rep) if k != 'conditions'})[:300]}", flush=True)
    print("[judge-s3] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
