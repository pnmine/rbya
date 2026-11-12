"""A small discrete-time simulator of a microservice shop.

One tick stands for 30 seconds. Faults are injected into a service, node or
the database, and the simulator produces three kinds of telemetry every tick:
metrics, log lines and sampled traces. It also keeps the ground truth (root
cause and correct fix) so runs can be scored automatically.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .actions import Action

TICK_SECONDS = 30

SERVICES = ["frontend", "cart", "catalog", "checkout", "payment", "inventory", "db"]
DEPS = {
    "frontend": ["cart", "catalog", "checkout"],
    "checkout": ["payment", "inventory", "cart"],
    "cart": ["db"],
    "catalog": ["db"],
    "inventory": ["db"],
    "payment": [],
    "db": [],
}
NODES = {
    "node-a": ["frontend", "cart", "catalog"],
    "node-b": ["checkout", "payment", "inventory"],
    "node-c": ["db"],
}
APP_SERVICES = [s for s in SERVICES if s != "db"]

BASE_LATENCY = {"frontend": 120, "cart": 60, "catalog": 50, "checkout": 140,
                "payment": 90, "inventory": 45, "db": 15}

FAULT_KINDS = ["bad_deploy", "config_error", "memory_leak", "traffic_spike",
               "transient", "slow_query", "shifting", "node_pressure"]


@dataclass
class Fault:
    kind: str
    target: str
    start: int = 0
    active: bool = True
    age: int = 0
    data: dict = field(default_factory=dict)

    def correct_fix(self) -> Action | None:
        return {
            "bad_deploy": Action("rollback_deployment", self.target),
            "shifting": Action("rollback_deployment", self.target),
            "config_error": Action("revert_config", self.target),
            "memory_leak": Action("restart_pod", self.target),
            "traffic_spike": Action("scale_replicas", self.target),
            "slow_query": Action("kill_slow_query", "db"),
            "node_pressure": Action("drain_node", self.target),
            "transient": None,
        }[self.kind]


@dataclass
class ServiceState:
    version: int = 1
    prev_version: int = 1
    config_rev: int = 1
    replicas: int = 2
    mem: float = 0.35
    restarts: int = 0
    disruption: int = 0          # ticks of self-inflicted disruption left


NOISE_LINES = [
    "[{s}] WARN GC pause 410ms",
    "[{s}] WARN retrying request to {d} (attempt 2)",
    "[{s}] INFO cache miss ratio 0.31",
    "[{s}] WARN slow response from {d} (1.2s)",
    "[{s}] ERROR client closed connection early",
]


class Simulator:
    def __init__(self, fault: Fault, seed: int = 0, noise: float = 0.0):
        self.rng = random.Random(seed)
        self.noise_rng = random.Random(seed + 7919)
        self.noise = noise          # 0 = clear evidence; 1 = red herrings every tick
        self.t = 0
        self.fault = fault
        self.svc = {s: ServiceState() for s in SERVICES}
        self.db_outage = 0
        self.drained: set[str] = set()
        self.events: list[str] = []          # one-off log lines (deploys, config changes)
        self.collateral = 0                  # ticks of damage caused by actions
        self.action_log: list[tuple[int, str, bool]] = []
        self.clear_streak = 0
        self.last_obs = None

    # ------------------------------------------------------------- faults
    def _inject(self):
        f = self.fault
        if f.kind in ("bad_deploy", "shifting"):
            s = self.svc[f.target]
            s.prev_version, s.version = s.version, s.version + 1
            self.events.append(f"[{f.target}] deployment updated to v{s.version}")
        elif f.kind == "config_error":
            self.svc[f.target].config_rev += 1
            self.events.append(f"[{f.target}] configmap {f.target}-env updated (rev {self.svc[f.target].config_rev})")

    def _fault_effects(self):
        """Return per-service (extra_error, latency_factor, cpu, mem, log lines)."""
        f = self.fault
        err = {s: 0.0 for s in SERVICES}
        lat = {s: 1.0 for s in SERVICES}
        cpu = {s: 0.3 for s in SERVICES}
        logs: list[str] = []
        if not f.active:
            return err, lat, cpu, logs
        a = f.age
        if f.kind == "bad_deploy":
            err[f.target] = 0.32
            logs.append(f"[{f.target}] ERROR NullPointerException in OrderHandler.process (v{self.svc[f.target].version})")
        elif f.kind == "config_error":
            err[f.target] = 0.45
            logs.append(f"[{f.target}] ERROR connection refused: upstream host not found (DB_HOST)")
        elif f.kind == "memory_leak":
            s = self.svc[f.target]
            s.mem = min(0.99, s.mem + 0.09)
            lat[f.target] = 1.0 + 3.0 * max(0.0, s.mem - 0.6)
            if s.mem > 0.9:
                s.restarts += 1
                err[f.target] = 0.15
                logs.append(f"[{f.target}] WARN container OOMKilled, restarting")
            elif s.mem > 0.7:
                logs.append(f"[{f.target}] WARN heap usage {int(s.mem * 100)}%")
        elif f.kind == "traffic_spike":
            factor = 1.0 / max(1, self.svc[f.target].replicas - 1)
            cpu[f.target] = min(0.99, 0.6 + 0.4 * factor)
            lat[f.target] = 1.0 + 9.0 * factor
            logs.append(f"[{f.target}] WARN request queue length {int(400 * factor)}")
        elif f.kind == "transient":
            if a <= 2:
                err[f.target] = 0.12
                lat[f.target] = 4.0
                logs.append(f"[{f.target}] WARN upstream timeout (retrying)")
            else:
                f.active = False
        elif f.kind == "slow_query":
            lat["db"] = 40.0
            cpu["db"] = 0.97
            for s in ("cart", "catalog", "inventory"):
                lat[s] = 6.0
                err[s] = 0.06
            logs.append("[db] WARN slow query: SELECT * FROM orders JOIN items ... (running 412s)")
            logs.append("[cart] WARN db call exceeded 2s")
        elif f.kind == "shifting":
            # Early on the clearest symptom is latency at the frontend; the real
            # errors at the target grow over the next ticks.
            grow = min(1.0, a / 3.0)
            lat["frontend"] = 7.0
            err[f.target] = 0.04 + 0.36 * grow
            if a >= 2:
                logs.append(f"[{f.target}] ERROR TimeoutException calling card gateway (v{self.svc[f.target].version})")
            logs.append("[frontend] WARN slow responses from checkout")
        elif f.kind == "node_pressure":
            for s in NODES[f.target]:
                err[s] = 0.2
                lat[s] = 3.0
            logs.append(f"[{f.target}] WARN DiskPressure: evicting pods")
            for s in NODES[f.target]:
                logs.append(f"[{s}] WARN pod evicted (node {f.target} DiskPressure)")
        return err, lat, cpu, logs

    # ------------------------------------------------------------- actions
    def apply(self, action: Action) -> bool:
        """Apply an action. Returns True if it removed the active fault."""
        f = self.fault
        fixed = False
        n, tgt = action.name, action.target
        if n == "restart_pod":
            s = self.svc[tgt]
            s.restarts += 1
            s.disruption = max(s.disruption, 1)
            if f.active and f.kind == "memory_leak" and f.target == tgt:
                s.mem = 0.35
                fixed = True
        elif n == "scale_replicas":
            self.svc[tgt].replicas += 2
            if f.active and f.kind == "traffic_spike" and f.target == tgt:
                fixed = True
        elif n == "rollback_deployment":
            s = self.svc[tgt]
            if s.version != s.prev_version:
                s.version, s.prev_version = s.prev_version, s.version
                self.events.append(f"[{tgt}] rolled back to v{s.version}")
            if f.active and f.kind in ("bad_deploy", "shifting") and f.target == tgt:
                fixed = True
        elif n == "revert_config":
            if f.active and f.kind == "config_error" and f.target == tgt:
                fixed = True
        elif n == "kill_slow_query":
            if f.active and f.kind == "slow_query":
                fixed = True
        elif n == "restart_database":
            self.db_outage = 3
            self.collateral += 3
            if f.active and f.kind == "slow_query":
                fixed = True
        elif n == "drain_node":
            self.drained.add(tgt)
            if f.active and f.kind == "node_pressure" and f.target == tgt:
                fixed = True
            else:
                for s in NODES[tgt]:
                    self.svc[s].disruption = max(self.svc[s].disruption, 2)
                self.collateral += 2
        if fixed:
            f.active = False
        self.action_log.append((self.t, str(action), fixed))
        return fixed

    def undo(self, action: Action) -> None:
        if action.name == "rollback_deployment":
            s = self.svc[action.target]
            s.version, s.prev_version = s.prev_version, s.version
            self.events.append(f"[{action.target}] redeployed v{s.version}")
        elif action.name == "scale_replicas":
            self.svc[action.target].replicas = max(2, self.svc[action.target].replicas - 2)
        # revert_config undo is a no-op in this model

    # ------------------------------------------------------------- step
    def step(self) -> dict:
        if self.t == self.fault.start:
            self._inject()
        if self.t >= self.fault.start:
            self.fault.age = self.t - self.fault.start
        err, lat, cpu, fault_logs = self._fault_effects() if self.t >= self.fault.start else (
            {s: 0.0 for s in SERVICES}, {s: 1.0 for s in SERVICES}, {s: 0.3 for s in SERVICES}, [])

        logs = list(self.events) + fault_logs
        self.events = []
        if self.noise > 0 and self.t >= self.fault.start:
            nr = self.noise_rng
            if self.t == self.fault.start and nr.random() < self.noise:
                # an unrelated, harmless deploy right around the time the fault starts
                others = [x for x in APP_SERVICES if x != self.fault.target]
                victim = nr.choice(others)
                st_ = self.svc[victim]
                st_.prev_version, st_.version = st_.version, st_.version + 1
                logs.append(f"[{victim}] deployment updated to v{st_.version}")
            for _ in range(2):
                if nr.random() < self.noise:
                    src = nr.choice(APP_SERVICES)
                    dst = nr.choice(DEPS[src] or ["db"])
                    logs.append(nr.choice(NOISE_LINES).format(s=src, d=dst))
        if self.db_outage > 0:
            err["db"] = 1.0
            for s in ("cart", "catalog", "inventory"):
                err[s] = max(err[s], 0.5)
            logs.append("[db] ERROR database restarting, connections refused")
            self.db_outage -= 1
        for s, st in self.svc.items():
            if st.disruption > 0:
                err[s] = max(err[s], 0.25)
                logs.append(f"[{s}] WARN pods not ready (restarting)")
                st.disruption -= 1

        # Errors propagate to callers at a reduced rate.
        prop = dict(err)
        for _ in range(2):
            for caller, deps in DEPS.items():
                for d in deps:
                    prop[caller] = max(prop[caller], 0.5 * prop[d])

        metrics = {}
        for s in SERVICES:
            noise = self.rng.uniform(0.9, 1.1)
            down_lat = max([lat[d] for d in DEPS[s]] + [1.0])
            latf = max(lat[s], 1.0 + 0.6 * (down_lat - 1.0))
            metrics[s] = {
                "p99_ms": round(BASE_LATENCY[s] * latf * noise),
                "error_rate": round(min(1.0, prop[s] + self.rng.uniform(0, 0.005)), 3),
                "cpu": round(min(0.99, cpu[s] * self.rng.uniform(0.95, 1.05)), 2),
                "mem": round(self.svc[s].mem, 2),
                "restarts": self.svc[s].restarts,
                "version": self.svc[s].version,
                "replicas": self.svc[s].replicas,
            }

        traces = []
        for s in SERVICES:
            for d in DEPS[s]:
                if err[d] > 0.03:
                    traces.append(f"span {s} -> {d}: status=ERROR ({d} returned 5xx)")
                elif lat[d] > 2.5:
                    traces.append(f"span {s} -> {d}: duration={int(BASE_LATENCY[d] * lat[d])}ms (slow)")

        alerts = []
        for s, m in metrics.items():
            if m["error_rate"] > 0.05:
                alerts.append((s, "HighErrorRate"))
            if m["p99_ms"] > 3 * BASE_LATENCY[s]:
                alerts.append((s, "HighLatency"))
            if m["mem"] > 0.85:
                alerts.append((s, "HighMemory"))
        alerts.sort()

        obs = {"t": self.t, "metrics": metrics, "logs": logs, "traces": sorted(set(traces)),
               "alerts": alerts}
        self.clear_streak = self.clear_streak + 1 if not alerts else 0
        self.last_obs = obs
        self.t += 1
        return obs

    def resolved(self) -> bool:
        return (not self.fault.active) and self.clear_streak >= 2


def sample_fault(rng: random.Random, kind: str | None = None) -> Fault:
    kind = kind or rng.choice(FAULT_KINDS)
    if kind == "node_pressure":
        target = rng.choice(["node-a", "node-b"])
    elif kind == "slow_query":
        target = "db"
    elif kind == "shifting":
        target = rng.choice(["payment", "inventory"])
    else:
        target = rng.choice(APP_SERVICES)
    return Fault(kind=kind, target=target, start=1)
