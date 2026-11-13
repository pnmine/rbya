"""Agents compared in the study.

ReActAgent        - B1: proposes one action and runs it straight away.
RollbackAgent     - B2: runs the action, rolls it back if a post-check fails
                    (the binary safety idea behind STRATUS-style agents).
ReconsiderAgent   - ours: candidate -> commit by risk tier -> reconsider before
                    acting, plus knowledge compilation of verified fixes.
                    With compile=False it is the "ours without compilation" arm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .actions import HIGH, LOW, MEDIUM, Action
from .human import SimulatedHuman
from .memory import ProcedureStore
from .sim import TICK_SECONDS, Simulator
from .telemetry import alerting_services, evidence_key


@dataclass
class IncidentResult:
    agent: str
    fault: str
    target: str
    resolved: bool = False
    ttr_min: float | None = None
    executed: int = 0
    unnecessary: int = 0
    reversed: int = 0
    abandoned: int = 0
    escalations: int = 0
    rejected_by_human: int = 0
    harmful_high_risk: int = 0
    collateral_ticks: int = 0
    llm_calls: int = 0
    llm_seconds: float = 0.0
    llm_unparsed: int = 0
    store_hits: int = 0
    trace: list[str] = field(default_factory=list)


class BaseAgent:
    name = "base"

    def __init__(self, reasoner, settle: int = 2, max_ticks: int = 40, verbose: bool = False):
        self.reasoner = reasoner
        self.settle = settle
        self.max_ticks = max_ticks
        self.verbose = verbose

    # ----------------------------------------------------------- helpers
    def log(self, msg: str):
        self.res.trace.append(f"t={self.sim.t:>2} {msg}")
        if self.verbose:
            print(self.res.trace[-1])

    def tick(self) -> dict:
        self.obs = self.sim.step()
        return self.obs

    def wait(self, n: int) -> dict:
        for _ in range(n):
            self.tick()
            if self.done():
                break
        return self.obs

    def done(self) -> bool:
        return self.sim.resolved() or self.sim.t > self.detect_t + self.max_ticks

    def propose(self, exclude):
        before, t_before = self.reasoner.calls, getattr(self.reasoner, "model_seconds", 0.0)
        prop = self.reasoner.propose(self.obs, exclude)
        self.res.llm_calls += self.reasoner.calls - before
        self.res.llm_seconds += getattr(self.reasoner, "model_seconds", 0.0) - t_before
        if prop.candidates:
            top = prop.candidates[0]
            self.log(f"REASON {prop.diagnosis} | top: {top.action} ({top.confidence:.2f}) {top.rationale}")
        if not prop.candidates and prop.diagnosis == "unparseable reply":
            self.res.llm_unparsed += 1
            self.log("REASONER reply could not be parsed, waiting one tick")
        return prop

    def execute(self, action: Action) -> bool:
        fixed = self.sim.apply(action)
        self.res.executed += 1
        if not fixed:
            self.res.unnecessary += 1
            if action.tier == HIGH:
                self.res.harmful_high_risk += 1
        self.log(f"EXECUTE {action} [{action.tier}] -> {'fixed root cause' if fixed else 'no fix'}")
        return fixed

    # ----------------------------------------------------------- main
    def run(self, sim: Simulator) -> IncidentResult:
        self.sim = sim
        self.res = IncidentResult(self.name, sim.fault.kind, sim.fault.target)
        self.detect_t = 10 ** 6
        self.tick()
        while not self.obs["alerts"] and self.sim.t < 30:
            self.tick()
        self.detect_t = self.obs["t"]
        self.log(f"ALERT {sorted(alerting_services(self.obs))}")
        self.handle()
        if sim.resolved():
            self.res.resolved = True
            self.res.ttr_min = (self.obs["t"] - self.detect_t) * TICK_SECONDS / 60
        self.res.collateral_ticks = sim.collateral
        return self.res

    def handle(self):
        raise NotImplementedError


class ReActAgent(BaseAgent):
    name = "B1-ReAct"

    def handle(self):
        tried: set[str] = set()
        while not self.done():
            if not self.obs["alerts"]:
                self.tick()
                continue
            prop = self.propose(tried)
            if not prop.candidates:
                self.tick()
                continue
            a = prop.candidates[0].action
            if not self.execute(a):
                tried.add(str(a))
            self.wait(self.settle)


class RollbackAgent(BaseAgent):
    name = "B2-ActRollback"

    def handle(self):
        tried: set[str] = set()
        while not self.done():
            if not self.obs["alerts"]:
                self.tick()
                continue
            before = len(self.obs["alerts"])
            prop = self.propose(tried)
            if not prop.candidates:
                self.tick()
                continue
            a = prop.candidates[0].action
            self.execute(a)
            self.wait(self.settle)
            if self.obs["alerts"] and len(self.obs["alerts"]) >= before:
                tried.add(str(a))
                if a.spec.reversible:
                    self.sim.undo(a)
                    self.res.reversed += 1
                    self.log(f"POST-CHECK failed, ROLLBACK {a}")
                else:
                    self.log(f"POST-CHECK failed, {a} cannot be undone")


class ReconsiderAgent(BaseAgent):
    def __init__(self, reasoner, human: SimulatedHuman, compile: bool = True,
                 store: ProcedureStore | None = None, max_replans: int = 2,
                 min_confidence: float = 0.4, max_holds: int = 2,
                 use_tiers: bool = True, use_reconsider: bool = True, use_hold: bool = True,
                 name: str | None = None, **kw):
        super().__init__(reasoner, **kw)
        self.use_tiers, self.use_reconsider, self.use_hold = use_tiers, use_reconsider, use_hold
        self.min_confidence = min_confidence   # below this a candidate is not committed yet
        self.max_holds = max_holds
        self.human = human
        self.compile = compile
        self.store = store if store is not None else ProcedureStore()
        self.max_replans = max_replans
        self.name = name or ("Ours" if compile else "Ours-noCompile")

    def _reconsider(self, plan_alerts: set[str]) -> str:
        """Re-check the symptom one tick later. Returns 'go', 'cleared' or 'changed'."""
        self.tick()
        now = alerting_services(self.obs)
        if not now:
            return "cleared"
        if now != plan_alerts:
            return "changed"
        return "go"

    def handle(self):
        tried: set[str] = set()
        deferred: list[Action] = []      # high-risk actions a human turned down
        replans = holds = failed_since_reject = 0
        while not self.done():
            if not self.obs["alerts"]:
                self.tick()
                continue
            key = evidence_key(self.obs)
            stored = self.store.lookup(key) if self.compile else None
            if deferred and failed_since_reject >= 2:
                # Alternatives did not work: bring the rejected action back to the human once.
                action, source = deferred.pop(0), "escalation queue (second review)"
                tried.discard(str(action))
                failed_since_reject = 0
            elif stored is not None and str(stored) not in tried:
                action, source = stored, "procedure store"
                self.res.store_hits += 1
            else:
                prop = self.propose(tried)
                if not prop.candidates:
                    self.tick()
                    continue
                top = prop.candidates[0]
                if self.use_hold and top.confidence < self.min_confidence and holds < self.max_holds:
                    holds += 1
                    self.log(f"HOLD: best candidate {top.action} has low confidence ({top.confidence:.2f}), gather more evidence")
                    self.tick()
                    continue
                action, source = top.action, "reasoner"
            holds = 0
            self.log(f"CANDIDATE {action} [{action.tier}] from {source}")

            # Reconsider before acting.
            verdict = (self._reconsider(alerting_services(self.obs))
                       if self.use_reconsider and replans < self.max_replans else "go")
            if verdict == "cleared":
                self.res.abandoned += 1
                self.log(f"RECONSIDER: alerts cleared, drop {action}")
                continue
            if verdict == "changed":
                self.res.abandoned += 1
                replans += 1
                self.log(f"RECONSIDER: symptom changed to {sorted(alerting_services(self.obs))}, drop {action} and re-plan")
                continue
            replans = 0

            # Commit by risk tier.
            if self.use_tiers and action.tier == HIGH:
                self.res.escalations += 1
                self.log(f"ESCALATE {action} to on-call human")
                self.wait(self.human.delay_ticks)
                if self.done():
                    break
                if not self.human.review(action, self.sim.fault.correct_fix() if self.sim.fault.active else None):
                    self.res.rejected_by_human += 1
                    tried.add(str(action))
                    if source != "escalation queue (second review)":
                        deferred.append(action)
                    failed_since_reject = 0
                    self.log(f"HUMAN rejected {action}")
                    continue
                self.log(f"HUMAN approved {action}")
            before = len(self.obs["alerts"])
            self.execute(action)
            self.wait(self.settle)
            ok = not self.obs["alerts"] or len(self.obs["alerts"]) < before
            if not ok:
                tried.add(str(action))
                failed_since_reject += 1
                if stored is not None and action == stored:
                    self.store.forget(key)
                if self.use_tiers and action.tier == MEDIUM and action.spec.reversible:
                    self.sim.undo(action)
                    self.res.reversed += 1
                    self.log(f"UNDO WINDOW: no improvement, undo {action}")
            elif self.compile and not self.obs["alerts"] and key:
                self.store.compile(key, action)
                self.log(f"COMPILE {key} -> {action}")
