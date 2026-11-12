"""Run all agent arms on the same incident sequences and write results.

Usage:
  python run_experiments.py --reasoner scripted --episodes 30 --out results/scripted
  python run_experiments.py --reasoner llm --episodes 5 --out results/llm   # needs RBYA_LLM_* env vars
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics as st
from collections import defaultdict
from dataclasses import asdict

from rbya.agents import ReActAgent, ReconsiderAgent, RollbackAgent
from rbya.human import SimulatedHuman
from rbya.memory import ProcedureStore
from rbya.reasoners import LLMReasoner, ScriptedReasoner
from rbya.sim import FAULT_KINDS, Simulator, sample_fault

ARMS = ["B1-ReAct", "B2-ActRollback", "Ours-noCompile", "Ours"]
ABLATIONS = ["Abl-tiersOnly", "Abl-noReconsider", "Abl-noHold", "Abl-noTiers"]


def incident_sequence(rng: random.Random, n_distinct: int, repeats: int):
    kinds = (FAULT_KINDS * ((n_distinct // len(FAULT_KINDS)) + 1))[:n_distinct]
    rng.shuffle(kinds)
    distinct = [sample_fault(rng, k) for k in kinds]
    seq = []
    for f in distinct:
        for _ in range(repeats):
            seq.append((f.kind, f.target))
    rng.shuffle(seq)
    occurrence = defaultdict(int)
    out = []
    for kind, target in seq:
        occurrence[(kind, target)] += 1
        out.append((kind, target, occurrence[(kind, target)]))
    return out


def make_reasoner(kind: str, seed: int, slip: float):
    return ScriptedReasoner(seed=seed, slip=slip)


def make_agent(arm, reasoner, seed, store):
    human = SimulatedHuman(seed=seed)
    if arm == "B1-ReAct":
        return ReActAgent(reasoner)
    if arm == "B2-ActRollback":
        return RollbackAgent(reasoner)
    if arm == "Ours-noCompile":
        return ReconsiderAgent(reasoner, human, compile=False)
    if arm == "Abl-tiersOnly":
        return ReconsiderAgent(reasoner, human, compile=False, use_reconsider=False, use_hold=False, name=arm)
    if arm == "Abl-noReconsider":
        return ReconsiderAgent(reasoner, human, compile=True, store=store, use_reconsider=False, name=arm)
    if arm == "Abl-noHold":
        return ReconsiderAgent(reasoner, human, compile=True, store=store, use_hold=False, name=arm)
    if arm == "Abl-noTiers":
        return ReconsiderAgent(reasoner, human, compile=True, store=store, use_tiers=False, name=arm)
    return ReconsiderAgent(reasoner, human, compile=True, store=store)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--distinct", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--slip", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--out", default="results/scripted")
    ap.add_argument("--save-traces", action="store_true")
    ap.add_argument("--noise", type=float, default=0.0, help="red-herring logs: 0 clear, 0.5 noisy")
    ap.add_argument("--model", default=None, help="LLM model name (overrides RBYA_LLM_MODEL)")
    ap.add_argument("--no-cache", action="store_true", help="do not reuse identical LLM replies")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = args.arms.split(",")

    llm = None
    if args.reasoner == "llm":
        llm = LLMReasoner(model=args.model,
                          cache_path=None if args.no_cache else os.path.join(args.out, "llm_cache.jsonl"),
                          log_path=os.path.join(args.out, "llm_log.jsonl"))
        print(f"LLM: {llm.model} at {llm.base_url}")

    # Results are appended as each incident finishes, so an interrupted run can resume.
    csv_path = os.path.join(args.out, "incidents.csv")
    rows, traces, done = [], [], set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("resolved", "top1_changed"):
                    if k in r:
                        r[k] = r[k] == "True"
                for k in ("episode", "index", "occurrence", "executed", "unnecessary", "reversed", "abandoned",
                          "escalations", "rejected_by_human", "harmful_high_risk", "collateral_ticks",
                          "llm_calls", "store_hits", "llm_unparsed"):
                    r[k] = int(r.get(k) or 0)
                r["ttr_min"] = float(r["ttr_min"]) if r["ttr_min"] else None
                r["llm_seconds"] = float(r.get("llm_seconds") or 0)
                rows.append(r)
                done.add((r["episode"], r["agent"]))
        print(f"resuming: {len(done)} (episode, arm) pairs already done")
    writer_fh = open(csv_path, "a", newline="")
    writer = None

    import time as _time
    for ep in range(args.episodes):
        t_ep = _time.time()
        seq = incident_sequence(random.Random(args.seed * 1000 + ep), args.distinct, args.repeats)
        for arm in arms:
            if (ep, arm) in done:
                continue
            reasoner = llm if llm is not None else make_reasoner(args.reasoner, seed=ep, slip=args.slip)
            store = ProcedureStore()
            arm_rows = []
            for i, (kind, target, occ) in enumerate(seq):
                fault = sample_fault(random.Random(0), kind)
                fault.target = target
                sim = Simulator(fault, seed=ep * 100 + i, noise=args.noise)
                agent = make_agent(arm, reasoner, seed=ep * 100 + i, store=store)
                res = agent.run(sim)
                row = {k: v for k, v in asdict(res).items() if k != "trace"}
                row.update(episode=ep, index=i, occurrence=occ)
                arm_rows.append(row)
                if args.save_traces:
                    traces.append({"episode": ep, "index": i, "agent": arm, "fault": kind,
                                    "target": target, "trace": res.trace})
            # write an arm only once all its incidents in this episode are done
            if writer is None:
                writer = csv.DictWriter(writer_fh, fieldnames=list(arm_rows[0].keys()))
                if writer_fh.tell() == 0:
                    writer.writeheader()
            writer.writerows(arm_rows)
            writer_fh.flush()
            rows.extend(arm_rows)
        msg = f"episode {ep + 1}/{args.episodes} done in {(_time.time() - t_ep) / 60:.1f} min"
        if llm is not None:
            msg += (f" | api calls {llm.api_calls}, cache hits {llm.cache_hits}, repairs {llm.repairs}, "
                    f"truncated {llm.truncated}, unparseable {llm.parse_failures}, failures {llm.failures}, "
                    f"waiting {llm.latency_total / 60:.1f} min")
        print(msg, flush=True)
    writer_fh.close()

    if traces:
        with open(os.path.join(args.out, "traces.jsonl"), "a") as fh:
            for t in traces:
                fh.write(json.dumps(t) + "\n")
    summary = summarize(rows, arms)
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        extra = {}
        if llm is not None:
            extra = {"model": llm.model, "api_calls": llm.api_calls, "cache_hits": llm.cache_hits,
                     "repairs": llm.repairs, "truncated": llm.truncated, "max_tokens": llm.max_tokens,
                     "failures": llm.failures, "unparseable": llm.parse_failures,
                     "seconds_waiting": round(llm.latency_total, 1),
                     "mean_seconds_per_call": round(llm.latency_total / max(1, llm.api_calls), 2)}
        json.dump({"args": vars(args), "llm": extra, "summary": summary}, fh, indent=2)
    print_table(summary, arms)


def summarize(rows, arms):
    out = {}
    for arm in arms:
        r = [x for x in rows if x["agent"] == arm]
        n = len(r)
        ttr = [x["ttr_min"] for x in r if x["resolved"]]
        first = [x["ttr_min"] for x in r if x["resolved"] and x["occurrence"] == 1]
        rep = [x["ttr_min"] for x in r if x["resolved"] and x["occurrence"] > 1]
        out[arm] = {
            "incidents": n,
            "resolution_rate": sum(x["resolved"] for x in r) / n,
            "mttr_min": st.mean(ttr) if ttr else None,
            "mttr_first_min": st.mean(first) if first else None,
            "mttr_repeat_min": st.mean(rep) if rep else None,
            "unnecessary_per_incident": sum(x["unnecessary"] for x in r) / n,
            "reversed_per_incident": sum(x["reversed"] for x in r) / n,
            "abandoned_per_incident": sum(x["abandoned"] for x in r) / n,
            "escalations_per_incident": sum(x["escalations"] for x in r) / n,
            "harmful_high_risk_total": sum(x["harmful_high_risk"] for x in r),
            "collateral_ticks_total": sum(x["collateral_ticks"] for x in r),
            "llm_calls_per_incident": sum(x["llm_calls"] for x in r) / n,
            "store_hits_per_incident": sum(x["store_hits"] for x in r) / n,
            "llm_seconds_per_incident": sum(x.get("llm_seconds", 0) for x in r) / n,
            "llm_unparsed_per_incident": sum(x.get("llm_unparsed", 0) for x in r) / n,
        }
    return out


def print_table(summary, arms):
    keys = ["resolution_rate", "mttr_min", "mttr_first_min", "mttr_repeat_min", "unnecessary_per_incident",
            "reversed_per_incident", "abandoned_per_incident", "escalations_per_incident",
            "harmful_high_risk_total", "collateral_ticks_total", "llm_calls_per_incident"]
    print("metric".ljust(28) + "".join(a.rjust(16) for a in arms))
    for k in keys:
        vals = []
        for a in arms:
            v = summary[a][k]
            vals.append(("-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)).rjust(16))
        print(k.ljust(28) + "".join(vals))


if __name__ == "__main__":
    main()
