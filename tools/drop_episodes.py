"""Remove episodes from a results folder so they can be run again.

    python -m tools.drop_episodes results/llm_noisy_7b --keep 0          # keep only episode 0
    python -m tools.drop_episodes results/llm_noisy_7b --drop 3 4        # drop episodes 3 and 4
"""
import argparse
import csv
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("folder")
g = ap.add_mutually_exclusive_group(required=True)
g.add_argument("--keep", type=int, nargs="+")
g.add_argument("--drop", type=int, nargs="+")
a = ap.parse_args()
want = (lambda e: e in a.keep) if a.keep is not None else (lambda e: e not in a.drop)

path = os.path.join(a.folder, "incidents.csv")
rows = list(csv.DictReader(open(path, newline="")))
kept = [r for r in rows if want(int(r["episode"]))]
with open(path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(kept)
print(f"incidents.csv: {len(rows)} -> {len(kept)} rows")

tp = os.path.join(a.folder, "traces.jsonl")
if os.path.exists(tp):
    lines = [l for l in open(tp, encoding="utf-8") if l.strip()]
    keep_l = [l for l in lines if want(json.loads(l)["episode"])]
    open(tp, "w", encoding="utf-8").writelines(keep_l)
    print(f"traces.jsonl: {len(lines)} -> {len(keep_l)} lines")
sp = os.path.join(a.folder, "summary.json")
if os.path.exists(sp):
    os.remove(sp)
    print("summary.json removed (the next run writes a new one)")
