"""Reasoners propose a diagnosis and a ranked list of candidate actions.

ScriptedReasoner is a rule-based stand-in for an LLM with a tunable chance of
following the surface symptom instead of the evidence. LLMReasoner calls any
OpenAI-compatible chat endpoint (Ollama, vLLM, Gemini's compatibility API, ...).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .actions import CATALOG, Action, is_valid
from .sim import APP_SERVICES, DEPS, NODES, SERVICES
from .telemetry import render


@dataclass
class Candidate:
    action: Action
    confidence: float
    rationale: str


@dataclass
class Proposal:
    diagnosis: str
    candidates: list[Candidate] = field(default_factory=list)
    raw: str = ""


class ScriptedReasoner:
    """Mimics an LLM that reads logs well but sometimes trusts the loudest alert."""

    def __init__(self, seed: int = 0, slip: float = 0.25, risky: float = 0.4):
        self.rng = random.Random(seed)
        self.slip = slip        # chance of blaming the service with the loudest symptom
        self.risky = risky      # chance of preferring the heavier fix when two exist
        self.calls = 0
        self.latency_total = 0.0

    def _root(self, obs) -> str:
        m = obs["metrics"]
        alerting = {s for s, _ in obs["alerts"]} or set(SERVICES)
        loud = max(alerting, key=lambda s: (m[s]["error_rate"], m[s]["p99_ms"] / 100))
        if self.rng.random() < self.slip:
            return loud
        # Follow the evidence downstream: a service whose dependencies are healthy.
        cand = [s for s in alerting if not any(d in alerting for d in DEPS[s])]
        if cand:
            return max(cand, key=lambda s: (m[s]["error_rate"], m[s]["p99_ms"] / 100))
        return loud

    def propose(self, obs: dict, exclude: set[str] = frozenset()) -> Proposal:
        self.calls += 1
        logs = "\n".join(obs["logs"])
        root = self._root(obs)
        sloppy = self.rng.random() < self.slip   # jumps to a generic fix without reading closely
        cands: list[Candidate] = []

        def add(name, target, conf, why):
            a = Action(name, target)
            if any(c.action == a for c in cands):
                return
            if str(a) not in exclude and is_valid(name, target, SERVICES, NODES):
                cands.append(Candidate(a, conf, why))

        node = next((n for n in NODES if f"[{n}] WARN DiskPressure" in logs), None)
        if node:
            add("drain_node", node, 0.8, f"{node} reports DiskPressure and is evicting pods")
            for s in NODES[node]:
                add("restart_pod", s, 0.3, "pods were evicted")
        if "[db] WARN slow query" in logs:
            first, second = ("restart_database", "kill_slow_query") if self.rng.random() < self.risky \
                else ("kill_slow_query", "restart_database")
            add(first, "db", 0.7, "a long-running query is holding the database")
            add(second, "db", 0.5, "a long-running query is holding the database")
        if root in APP_SERVICES:
            if f"[{root}] deployment updated" in logs or re.search(rf"\[{root}\] ERROR .*Exception", logs):
                add("rollback_deployment", root, 0.75, f"{root} errors started after a new version")
            if f"[{root}] configmap" in logs or f"[{root}] ERROR connection refused" in logs:
                add("revert_config", root, 0.75, f"{root} cannot reach a dependency after a config change")
            if f"[{root}] WARN container OOMKilled" in logs or f"[{root}] WARN heap usage" in logs:
                add("restart_pod", root, 0.7, f"{root} memory keeps growing")
            if f"[{root}] WARN request queue" in logs:
                add("scale_replicas", root, 0.7, f"{root} is CPU bound with a long queue")
            add("restart_pod", root, 0.85 if sloppy else 0.35, f"generic restart of {root}")
            add("rollback_deployment", root, 0.25, f"{root} might have a bad version")
        if not cands:
            for s in sorted({s for s, _ in obs["alerts"]} & set(APP_SERVICES)):
                add("restart_pod", s, 0.2, "no clear cause, try a restart")
        if sloppy:
            for c in cands:
                if c.action.name != "restart_pod":
                    c.confidence = round(c.confidence * 0.6, 2)
        cands.sort(key=lambda c: -c.confidence)
        return Proposal(diagnosis=f"suspected root: {root}", candidates=cands[:3])


SYSTEM_PROMPT = """You are an on-call SRE agent for a microservice shop.
Services and dependencies: {deps}
Nodes: {nodes}
Allowed actions (name: what it does, target type):
{actions}
Read the evidence and answer ONLY with JSON:
{{"diagnosis": "<one sentence>", "candidates": [{{"action": "<name>", "target": "<service|node|db>", "confidence": <0-1>, "rationale": "<short>"}}]}}
Give at most 3 candidates, best first. Do not repeat these already rejected actions: {exclude}"""


def _ssl_context() -> ssl.SSLContext:
    """Certificate verification stays on by default.

    RBYA_LLM_CA_BUNDLE  path to a PEM file with extra CA certificates
    RBYA_LLM_INSECURE=1 skip verification (last resort, prints a warning)
    If the optional `truststore` package is installed, the OS certificate store is used.
    """
    if os.environ.get("RBYA_LLM_INSECURE") == "1":
        print("WARNING: TLS certificate verification is disabled (RBYA_LLM_INSECURE=1)")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    try:
        import truststore  # type: ignore
        ctx = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        ctx = ssl.create_default_context()
    cafile = os.environ.get("RBYA_LLM_CA_BUNDLE")
    if cafile:
        ctx.load_verify_locations(cafile)
    return ctx


def _extract_json(text: str) -> dict | None:
    """Find the first JSON object in the reply that has a "candidates" key.
    Thinking blocks and markdown fences are ignored."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("```json", "").replace("```", "")
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "candidates" in obj:
            return obj
    return None


def _confidence(v) -> float:
    try:
        f = float(v.strip().rstrip("%")) if isinstance(v, str) else float(v)
        if f > 1:
            f = f / 100
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, f))


class LLMReasoner:
    """Any OpenAI-compatible chat endpoint (CPE AI Gateway, Ollama, vLLM, ...).

    Environment variables: RBYA_LLM_BASE_URL, RBYA_LLM_MODEL, RBYA_LLM_API_KEY,
    RBYA_LLM_MAX_TOKENS (default 4096), RBYA_LLM_TIMEOUT (default 180 s),
    RBYA_LLM_LOG (jsonl file of every prompt and reply), RBYA_LLM_CACHE (jsonl reply cache).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0, timeout: int | None = None,
                 max_tokens: int | None = None, cache_path: str | None = None, log_path: str | None = None,
                 retries: int = 4):
        env = os.environ.get
        self.base_url = (base_url or env("RBYA_LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or env("RBYA_LLM_MODEL", "qwen2.5:7b-instruct")
        self.api_key = api_key or env("RBYA_LLM_API_KEY", "none")
        self.temperature = temperature
        self.timeout = timeout or int(env("RBYA_LLM_TIMEOUT", "180"))
        self.max_tokens = max_tokens or int(env("RBYA_LLM_MAX_TOKENS", "4096"))
        self.retries = retries
        self.cache_path = cache_path or env("RBYA_LLM_CACHE")
        self.log_path = log_path or env("RBYA_LLM_LOG")
        self.ctx = _ssl_context() if self.base_url.startswith("https") else None
        self.calls = 0              # proposals requested by the agent
        self.api_calls = 0          # requests that actually went to the server
        self.cache_hits = 0
        self.failures = 0
        self.parse_failures = 0
        self.latency_total = 0.0    # seconds spent waiting for the server
        self.model_seconds = 0.0    # server time per proposal, counting cached replies at their original cost
        self.repairs = 0
        self.truncated = 0
        self._cache: dict[str, tuple[str, float]] = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = (rec["raw"], float(rec.get("seconds", 0.0)))
                    except (json.JSONDecodeError, KeyError):
                        pass

    def _post(self, system: str, user: str, history: list | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}] + (history or [])
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens,
                   "messages": messages}
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key}"})
        last = None
        for attempt in range(self.retries):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self.ctx) as r:
                    data = json.loads(r.read())
                self.latency_total += time.time() - t0
                self.api_calls += 1
                choice = data["choices"][0]
                msg = choice["message"]
                if choice.get("finish_reason") == "length":
                    self.truncated += 1
                content = msg.get("content") or ""
                if not content.strip():
                    content = msg.get("reasoning_content") or msg.get("reasoning") or ""
                return content
            except urllib.error.HTTPError as e:
                self.latency_total += time.time() - t0
                last = f"HTTP {e.code}: {e.read()[:300]!r}"
                if e.code not in (408, 429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                self.latency_total += time.time() - t0
                last = repr(e)
                if isinstance(getattr(e, "reason", None), ssl.SSLError):
                    raise RuntimeError(f"TLS error talking to {self.base_url}: {e}. "
                                       "Try `pip install truststore` or set RBYA_LLM_CA_BUNDLE.") from e
            time.sleep(min(30, 2 ** attempt * 2))
        self.failures += 1
        print(f"LLM request failed after {self.retries} tries: {last}")
        return ""

    def _chat(self, system: str, user: str) -> str:
        key = hashlib.sha256(f"{self.model}\n{self.temperature}\n{system}\n{user}".encode()).hexdigest()
        if key in self._cache:
            self.cache_hits += 1
            raw, secs = self._cache[key]
            self.model_seconds += secs
            return raw
        t0 = time.time()
        raw = self._post(system, user)
        repaired = False
        if raw and _extract_json(raw) is None:
            # One repair turn: show the model its reply and ask for the JSON only.
            self.repairs += 1
            repaired = True
            fixed = self._post(system, user, history=[
                {"role": "assistant", "content": raw[-3000:]},
                {"role": "user", "content": "Your reply did not contain the required JSON object. "
                                            "Reply again with the JSON object only, no other text."}])
            if fixed:
                raw = fixed
        secs = time.time() - t0
        self.model_seconds += secs
        if raw and self.cache_path and _extract_json(raw) is not None:
            self._cache[key] = (raw, secs)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, "raw": raw, "seconds": round(secs, 2)}) + "\n")
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": self.model, "seconds": round(secs, 2), "repaired": repaired,
                                     "parsed": _extract_json(raw) is not None, "user": user, "raw": raw}) + "\n")
        return raw

    def propose(self, obs: dict, exclude: set[str] = frozenset()) -> Proposal:
        self.calls += 1
        actions = "\n".join(f"- {a.name}: {a.description} (target: {a.target_kind})" for a in CATALOG.values())
        system = SYSTEM_PROMPT.format(deps=json.dumps(DEPS), nodes=json.dumps(NODES),
                                      actions=actions, exclude=sorted(exclude) or "none")
        raw = self._chat(system, render(obs))
        prop = Proposal(diagnosis="unparseable reply", raw=raw)
        data = _extract_json(raw)
        if data is None:
            self.parse_failures += 1
            return prop
        prop.diagnosis = str(data.get("diagnosis", ""))
        for c in (data.get("candidates") or [])[:3]:
            if not isinstance(c, dict):
                continue
            name = str(c.get("action", "")).strip()
            target = str(c.get("target", "")).strip()
            if is_valid(name, target, SERVICES, NODES) and f"{name}({target})" not in exclude:
                prop.candidates.append(Candidate(Action(name, target), _confidence(c.get("confidence", 0.5)),
                                                 str(c.get("rationale", ""))))
        return prop
