"""Turn a raw observation into compact evidence for the reasoner."""
from __future__ import annotations


def signature(obs: dict) -> frozenset:
    """Alert signature used by the reconsider check and the procedure store."""
    return frozenset(obs["alerts"])


def alerting_services(obs: dict) -> set[str]:
    return {s for s, _ in obs["alerts"]}


def render(obs: dict, max_logs: int = 8, max_traces: int = 8) -> str:
    """Late fusion at the evidence level: each signal is summarised on its own."""
    lines = [f"Time: t={obs['t']}"]
    lines.append("ALERTS: " + (", ".join(f"{s}:{a}" for s, a in obs["alerts"]) or "none"))
    lines.append("METRICS (service: p99_ms, error_rate, cpu, mem, restarts, version, replicas):")
    for s, m in obs["metrics"].items():
        lines.append(f"  {s}: {m['p99_ms']}, {m['error_rate']}, {m['cpu']}, {m['mem']}, "
                     f"{m['restarts']}, v{m['version']}, {m['replicas']}")
    lines.append("LOGS:")
    lines += [f"  {l}" for l in obs["logs"][:max_logs]] or ["  (none)"]
    lines.append("TRACES:")
    lines += [f"  {t}" for t in obs["traces"][:max_traces]] or ["  (none)"]
    return "\n".join(lines)


def evidence_key(obs: dict) -> str | None:
    """A short fault signature for the procedure store: the most specific log
    keyword plus the service it came from. Returns None when nothing stands out."""
    keywords = [("OOMKilled", "oom"), ("heap usage", "oom"), ("NullPointerException", "exception"),
                ("TimeoutException", "exception"), ("connection refused: upstream", "config"),
                ("slow query", "slow_query"), ("DiskPressure: evicting", "disk"),
                ("request queue length", "queue")]
    for kw, tag in keywords:
        for line in obs["logs"]:
            if kw in line:
                where = line[1:line.index("]")]
                return f"{tag}@{where}"
    return None
