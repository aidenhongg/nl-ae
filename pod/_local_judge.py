"""Local primary-only judge for the OME Stage-2 PILOT.

Runs the design's PRIMARY judge (claude -p, the headline EM label) over the pod-generated
responses, on this machine where the claude CLI is available + authenticated. Parallelized with
resume via the judge cache; aggregated through behave.judge_all (cache hits => no duplicate claude
calls; secondary/safety judges are stubbed to None for the pilot — they are the full-grid robustness
judges, gated open models on the pod).

IMPORTANT — run from a NEUTRAL cwd with stdin closed so the nested claude is a clean rubric judge:
    cd /c/temp/omejudge && python <abs path>/_local_judge.py --workers 6 < /dev/null
(neutral cwd => the judge does not inherit this project's CLAUDE.md/MISSION context; stdin</dev/null
=> no 3s-per-call stdin wait and no context bleed.)
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
        for gp in sorted(C.PATHS.dir_behave().glob(f"gen_{d}_*.parquet")):
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
    ap.add_argument("--dirs", default="D_toxic,D_refusal,D_random_0")
    ap.add_argument("--sets", default="em")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()
    dirs = a.dirs.split(","); sets = tuple(a.sets.split(","))

    items = collect_items(dirs, sets)
    print(f"[judge] unique (prompt,response) to score: {len(items)}", flush=True)

    cache_path = C.PATHS.judge_cache()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r["role"] == "primary":
                    done.add((r["prompt_id"], r["response_sha"]))
    todo = [(pid, sha, pt, resp) for (pid, sha), (pt, resp) in items.items() if (pid, sha) not in done]
    print(f"[judge] cached={len(done)}  to-judge={len(todo)}  workers={a.workers}", flush=True)

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
                print(f"[judge] {n}/{len(todo)}  ok={counts['ok']} none={counts['none']}", flush=True)

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(work, todo))
    print(f"[judge] primary scoring done: ok={counts['ok']} none={counts['none']}", flush=True)

    # aggregate: cache pre-filled => judge_all does no new claude calls; open judges stubbed to None.
    primary_only = {"primary": lambda p, r: behave.judge_one(p, r),
                    "secondary": lambda p, r: None, "safety": lambda p, r: None}
    for d in dirs:
        if list(C.PATHS.dir_behave().glob(f"gen_{d}_*.parquet")):
            rep = behave.judge_all(d, sets=sets, judge_fns=primary_only)
            print(f"[judge] judge_all {d}: "
                  f"{json.dumps({k: rep[k] for k in list(rep) if k != 'conditions'})[:300]}", flush=True)
    print("[judge] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
