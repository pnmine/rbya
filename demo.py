"""Narrated walk-through of one incident, for the live demo.

  python demo.py --fault shifting --target payment
  python demo.py --fault slow_query --agent b1      # compare with the plain agent
  python demo.py --reasoner llm ...                 # use a real LLM (see README)
"""
import argparse
import os
import time

if os.name == "nt":
    os.system("")   # turn on ANSI colours in the Windows console

from rbya.agents import ReActAgent, ReconsiderAgent, RollbackAgent
from rbya.human import SimulatedHuman
from rbya.reasoners import LLMReasoner, ScriptedReasoner
from rbya.sim import FAULT_KINDS, Fault, Simulator
from rbya.telemetry import render

COLORS = {"ALERT": "\033[31m", "CANDIDATE": "\033[36m", "RECONSIDER": "\033[33m", "HOLD": "\033[33m",
          "ESCALATE": "\033[35m", "HUMAN": "\033[35m", "EXECUTE": "\033[32m", "UNDO": "\033[33m",
          "COMPILE": "\033[34m", "POST-CHECK": "\033[33m"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", choices=FAULT_KINDS, default="shifting")
    ap.add_argument("--target", default=None)
    ap.add_argument("--agent", choices=["ours", "b1", "b2"], default="ours")
    ap.add_argument("--reasoner", choices=["scripted", "llm"], default="scripted")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--slow", type=float, default=0.6, help="seconds to pause between lines")
    ap.add_argument("--show-evidence", action="store_true")
    args = ap.parse_args()

    default_target = {"slow_query": "db", "node_pressure": "node-b", "shifting": "payment"}.get(args.fault, "checkout")
    fault = Fault(args.fault, args.target or default_target, start=1)
    sim = Simulator(fault, seed=args.seed)
    reasoner = LLMReasoner() if args.reasoner == "llm" else ScriptedReasoner(seed=args.seed)
    agent = {"ours": lambda: ReconsiderAgent(reasoner, SimulatedHuman(seed=args.seed)),
             "b1": lambda: ReActAgent(reasoner), "b2": lambda: RollbackAgent(reasoner)}[args.agent]()

    print(f"\n=== Injected fault (hidden from the agent): {fault.kind} at {fault.target}; "
          f"correct fix: {fault.correct_fix()} ===\n")
    res = agent.run(sim)
    for line in res.trace:
        tag = line.split()[1] if len(line.split()) > 1 else ""
        tag = tag.split(":")[0]
        print(COLORS.get(tag, ""), line, "\033[0m", sep="")
        if args.show_evidence and tag == "ALERT":
            print(render(sim.last_obs))
        time.sleep(args.slow)
    print(f"\nResolved: {res.resolved}  time to recover: {res.ttr_min} min  actions run: {res.executed}  "
          f"unneeded: {res.unnecessary}  escalations: {res.escalations}  collateral ticks: {res.collateral_ticks}  "
          f"reasoner calls: {res.llm_calls}")


if __name__ == "__main__":
    main()
