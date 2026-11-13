"""RBYA dashboard: watch the agent handle an incident, step by step.

    pip install streamlit
    streamlit run dashboard.py

Two modes:
  Recorded run  replays an incident from an LLM experiment folder (results/llm_*),
                using the model's recorded replies from llm_cache.jsonl. Instant.
  Live run      injects a fault now. High-risk actions wait for you to approve them.
"""
from __future__ import annotations

import copy
import glob
import html
import os
import random
import re
import threading
import time

import streamlit as st

from rbya.agents import ReActAgent, ReconsiderAgent, RollbackAgent
from rbya.human import SimulatedHuman
from rbya.memory import ProcedureStore
from rbya.reasoners import LLMReasoner, ScriptedReasoner
from rbya.sim import DEPS, FAULT_KINDS, TICK_SECONDS, Fault, Simulator, sample_fault

st.set_page_config(page_title="RBYA: Reconsider Before You Act", layout="wide")

# ---------------------------------------------------------------- tokens (RBYA Blueprint design system)
T = dict(paper="#FDFDFE", surface="#FFFFFF", ink="#0032A0", deep="#0B2B86", muted="#4D67A2", line="#9DB1D9",
         grid="#DCE5F6", tint="#E3E9F3", focus="#1F5FE0", alert="#C2410C", alert_tint="#FCE8DD",
         pending="#8A5A00", pending_tint="#FBF0D9", resolved="#1D6B47", human="#6B3FA0", memory="#0F6E7A")
FONT = "https://cdn.jsdelivr.net/npm/@fontsource/iosevka@5.2.5"

CATEGORY = {  # trace tag -> (label shown, colour)
    "ALERT": ("Alert", T["alert"]), "REASON": ("Reasoning", T["ink"]), "CANDIDATE": ("Plan", T["ink"]),
    "REASONER": ("Reasoning", T["muted"]), "HOLD": ("Hold", T["pending"]), "RECONSIDER": ("Re-check", T["pending"]),
    "ESCALATE": ("Human", T["human"]), "HUMAN": ("Human", T["human"]), "EXECUTE": ("Action", T["focus"]),
    "UNDO": ("Undo", T["pending"]), "POST-CHECK": ("Post-check", T["pending"]), "COMPILE": ("Memory", T["memory"]),
}
LEGEND = {"Alert": T["alert"], "Plan": T["ink"], "Re-check": T["pending"], "Human": T["human"],
          "Action": T["focus"], "Memory": T["memory"]}
AGENTS = {"Ours": "Ours (re-check + risk tiers + memory)", "Ours-noCompile": "Ours without memory",
          "B1-ReAct": "B1 plain ReAct", "B2-ActRollback": "B2 act, then roll back"}
FAULT_NAMES = {"bad_deploy": "Bad deploy", "config_error": "Wrong config", "memory_leak": "Memory leak",
               "traffic_spike": "Traffic spike", "transient": "Brief blip (heals itself)",
               "slow_query": "Slow database query", "shifting": "Symptom moves to the real cause",
               "node_pressure": "Node out of disk"}


def css():
    st.markdown(f"""
<style>
@import url('{FONT}/400.css'); @import url('{FONT}/700.css'); @import url('{FONT}/800.css');
html, body, .stApp, button, input, select, textarea, p, span, div, label {{
  font-family: "Iosevka", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; }}
.stApp {{ background: {T['paper']}; color: {T['ink']}; }}
.block-container {{ padding: 1.2rem 1.5rem 1rem; max-width: 100%; }}
header[data-testid="stHeader"] {{ display: none; }}
p, label, span, div {{ color: {T['ink']}; }}
.stButton > button {{ border-radius: 0; border: 1.5px solid {T['ink']}; background: {T['surface']};
  color: {T['ink']}; font-weight: 700; letter-spacing: .04em; }}
.stButton > button:hover {{ background: {T['tint']}; color: {T['ink']}; border-color: {T['ink']}; }}
.stButton > button[kind="primary"] {{ background: {T['ink']}; }}
.stButton > button[kind="primary"] p {{ color: {T['surface']}; }}
.stButton > button:focus-visible {{ outline: 2px solid {T['focus']}; outline-offset: 2px; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ border-radius: 0 !important; }}
.panel {{ background: {T['surface']}; border: 1.5px solid {T['ink']}; padding: 12px 14px; margin-bottom: 12px; }}
.ptitle {{ font-size: 20px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; margin: 0 0 10px; }}
.apptitle {{ font-size: 27px; font-weight: 800; letter-spacing: .085em; text-transform: uppercase; line-height: 1.1; margin: 0; }}
.appsub {{ font-size: 13px; letter-spacing: .12em; text-transform: uppercase; color: {T['muted']} !important; margin: 2px 0 8px; }}
.muted {{ color: {T['muted']} !important; }}
.badge {{ display: inline-block; font-size: 12px; font-weight: 700; letter-spacing: .05em; padding: 2px 8px;
  border: 1px solid; border-radius: 2px; text-transform: uppercase; }}
.kv {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.kv td {{ border-top: 1px solid {T['line']}; padding: 5px 4px; vertical-align: top; }}
.kv td:first-child {{ font-weight: 700; letter-spacing: .04em; text-transform: uppercase; width: 42%; }}
.steps {{ list-style: none; margin: 0; padding: 0; max-height: 470px; overflow-y: auto; }}
.steps li {{ display: grid; grid-template-columns: 26px 1fr; gap: 8px; padding: 6px 0; border-top: 1px dashed {T['line']}; font-size: 13px; }}
.num {{ width: 22px; height: 22px; border-radius: 9999px; color: #fff !important; font-size: 12px; font-weight: 700;
  display: flex; align-items: center; justify-content: center; }}
.steps b {{ display: block; letter-spacing: .04em; text-transform: uppercase; font-size: 13px; }}
.log {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.log th {{ text-align: left; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; font-size: 12px;
  border-bottom: 1.5px solid {T['ink']}; padding: 4px 6px; }}
.log td {{ border-bottom: 1px solid {T['grid']}; padding: 4px 6px; vertical-align: top; }}
.log tr.now td {{ background: {T['tint']}; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 9999px; margin-right: 6px; }}
.approval {{ background: {T['pending_tint']}; border: 1.5px solid {T['pending']}; padding: 12px 14px; margin-bottom: 8px; }}
.approval .ptitle {{ color: {T['pending']} !important; }}
.approval p {{ color: {T['deep']}; font-size: 14px; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- running incidents
class CacheMiss(Exception):
    pass


class ReplayLLM(LLMReasoner):
    """Answers only from the recorded cache; never calls the server."""

    def _post(self, system, user, history=None):
        raise CacheMiss()


class ReplayReasoner:
    """Recorded LLM replies first; the scripted reasoner only for prompts that were never recorded."""

    def __init__(self, folder, seed):
        model = "muse-glimmer-30b"
        try:
            import json
            with open(os.path.join(folder, "summary.json"), encoding="utf-8") as fh:
                model = json.load(fh)["llm"]["model"]
        except Exception:
            pass
        self.model = model
        self.llm = ReplayLLM(base_url="http://recorded", model=model,
                             cache_path=os.path.join(folder, "llm_cache.jsonl"))
        self.fallback = ScriptedReasoner(seed=seed)
        self.calls, self.latency_total, self.model_seconds, self.misses = 0, 0.0, 0.0, 0

    def propose(self, obs, exclude=frozenset()):
        self.calls += 1
        try:
            before = self.llm.model_seconds
            prop = self.llm.propose(obs, exclude)
            self.model_seconds += self.llm.model_seconds - before
            return prop
        except CacheMiss:
            self.misses += 1
            return self.fallback.propose(obs, exclude)


class UIHuman:
    """The person at the dashboard approves or rejects high-risk actions."""

    def __init__(self, delay_ticks: int = 4):
        self.delay_ticks = delay_ticks
        self.requests = 0
        self.pending = None
        self.decision = None
        self.answered = threading.Event()

    def review(self, action, correct) -> bool:
        self.requests += 1
        self.decision = None
        self.answered.clear()
        self.pending = action
        self.answered.wait(timeout=900)
        self.pending = None
        return bool(self.decision)


def make_agent(arm, reasoner, human, store):
    if arm == "B1-ReAct":
        return ReActAgent(reasoner)
    if arm == "B2-ActRollback":
        return RollbackAgent(reasoner)
    return ReconsiderAgent(reasoner, human, compile=(arm == "Ours"), store=store)


def recorded_incident(sim, agent, frames):
    """Run one incident and keep a copy of what the agent saw at every tick."""
    orig = sim.step

    def step():
        obs = orig()
        frames.append(copy.deepcopy(obs))
        return obs

    sim.step = step
    return agent.run(sim)


TIER = {"low": "low risk", "medium": "medium risk", "high": "high risk"}


def plain(tag, rest):
    """Trace lines are written for logs; say the same thing in plain words for the page."""
    m = re.match(r"(\S+\(\S*\)) \[(\w+)\](.*)", rest)
    if tag == "ALERT":
        return "Alert on " + ", ".join(re.findall(r"'([^']+)'", rest))
    if tag == "CANDIDATE" and m:
        src = rest.split(" from ", 1)[-1]
        return f"Plan {m.group(1)}, {TIER.get(m.group(2), m.group(2))}, from the {src}"
    if tag == "EXECUTE" and m:
        return f"Ran {m.group(1)}. " + ("Fixed the cause." if "fixed root cause" in rest else "Did not fix the cause.")
    if tag == "RECONSIDER":
        if "cleared" in rest:
            return f"Re-checked: the alerts cleared on their own, so {rest.split('drop ')[-1]} was dropped"
        moved = ", ".join(re.findall(r"'([^']+)'", rest))
        return f"Re-checked: the symptom moved to {moved}. Dropped {rest.split('drop ')[-1].split(' and')[0]}, planning again"
    if tag == "HOLD":
        a = re.search(r"(\S+\(\S*\)) has low confidence \(([\d.]+)\)", rest)
        return f"Held {a.group(1)}: confidence only {a.group(2)}, waiting for more evidence" if a else rest
    if tag == "ESCALATE":
        return f"Asked the on-call engineer to approve {rest.split(' ')[0]}"
    if tag == "HUMAN":
        verb, act = rest.split(" ", 1)
        return f"Engineer {verb} {act}"
    if tag == "COMPILE":
        key, act = rest.split(" -> ")
        return f"Saved {act} as the fix for evidence {key}"
    if tag == "UNDO":
        return "No improvement in the undo window, undid " + rest.split("undo ")[-1]
    if tag == "POST-CHECK":
        return rest[0].upper() + rest[1:]
    if tag == "REASON":
        return rest.split(" | top:")[0]
    if tag == "REASONER":
        return "The model's reply could not be read, waiting one tick"
    return rest


def parse_events(trace):
    events = []
    for line in list(trace):
        m = re.match(r"t=\s*(\d+) (\S+) ?(.*)", line)
        if not m:
            continue
        t, tag, rest = int(m.group(1)), m.group(2).rstrip(":"), m.group(3)
        label, colour = CATEGORY.get(tag, (tag.title(), T["ink"]))
        events.append({"frame": max(0, t - 1), "tag": tag, "label": label, "colour": colour,
                       "text": plain(tag, rest), "raw": rest})
    return events


def replay(folder, episode, index, arm, noise, seed=7):
    """Re-run incident `index` of `episode` exactly as in the experiment, with recorded LLM replies.
    Earlier incidents of the episode run first so the procedure store is in the same state."""
    from run_experiments import incident_sequence
    seq = incident_sequence(random.Random(seed * 1000 + episode), 8, 2)
    reasoner = ReplayReasoner(folder, seed=episode)
    store = ProcedureStore()
    out = None
    for i, (kind, target, occ) in enumerate(seq[: index + 1]):
        fault = sample_fault(random.Random(0), kind)
        fault.target = target
        sim = Simulator(fault, seed=episode * 100 + i, noise=noise)
        agent = make_agent(arm, reasoner, SimulatedHuman(seed=episode * 100 + i), store)
        misses_before = reasoner.misses
        frames = []
        res = recorded_incident(sim, agent, frames)
        out = {"frames": frames, "res": res, "events": parse_events(res.trace), "fault": fault, "arm": arm,
               "occurrence": occ, "misses": reasoner.misses - misses_before,
               "source": f"{os.path.basename(folder)}, episode {episode}, incident {index}",
               "model": reasoner.model, "done": True}
    return out


def start_live(kind, target, arm, use_llm, noise):
    fault = Fault(kind, target, start=1)
    seed = int(time.time()) % 10000
    sim = Simulator(fault, seed=seed, noise=noise)
    reasoner = LLMReasoner() if use_llm else ScriptedReasoner(seed=seed)
    human = UIHuman()
    run = {"frames": [], "res": None, "events": [], "fault": fault, "arm": arm, "occurrence": 1, "misses": 0,
           "source": "live run", "model": getattr(reasoner, "model", "scripted reasoner"),
           "done": False, "human": human, "error": None}
    agent = make_agent(arm, reasoner, human, ProcedureStore())
    run["agent"] = agent

    def work():
        try:
            run["res"] = recorded_incident(sim, agent, run["frames"])
        except Exception as e:  # shown in the page
            run["error"] = repr(e)
        run["events"] = parse_events(agent.res.trace)
        run["done"] = True

    threading.Thread(target=work, daemon=True).start()
    return run


# ---------------------------------------------------------------- drawing
POS = {"frontend": (150, 230), "cart": (330, 120), "catalog": (330, 330), "checkout": (540, 330),
       "inventory": (720, 120), "payment": (720, 330), "db": (920, 225)}
NODE_BOX = {"node-a": (60, 50, 380, 370), "node-b": (450, 50, 380, 370), "node-c": (850, 50, 150, 370)}


def service_map(obs, last_target=None, drained=()):
    alerting = {s for s, _ in obs["alerts"]} if obs else set()
    m = obs["metrics"] if obs else {}
    W, H = 1020, 450
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Service map with alerting services in orange" '
           f'xmlns="http://www.w3.org/2000/svg" style="font-family:Iosevka,ui-monospace,monospace">',
           f'<defs><pattern id="g" width="16" height="16" patternUnits="userSpaceOnUse">'
           f'<path d="M16 0H0V16" fill="none" stroke="{T["grid"]}" stroke-width="1"/></pattern>'
           f'<marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0L10,5L0,10z" fill="{T["line"]}"/></marker>'
           f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
           f'<path d="M0,0L10,5L0,10z" fill="{T["alert"]}"/></marker></defs>',
           f'<rect width="{W}" height="{H}" fill="url(#g)"/>']
    for n, (x, y, w, h) in NODE_BOX.items():
        dash = "2 6" if n in drained else "6 5"
        label = f"{n} (drained)" if n in drained else n
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{T["line"]}" '
                   f'stroke-width="1.5" stroke-dasharray="{dash}"/>'
                   f'<text x="{x + 10}" y="{y + 22}" font-size="14" font-weight="700" fill="{T["muted"]}">{label}</text>')
    for src, deps in DEPS.items():
        for d in deps:
            (x1, y1), (x2, y2) = POS[src], POS[d]
            bad = d in alerting
            out.append(f'<line x1="{x1 + 50}" y1="{y1}" x2="{x2 - 52}" y2="{y2}" stroke="{T["alert"] if bad else T["line"]}" '
                       f'stroke-width="1.6" stroke-dasharray="6 5" marker-end="url(#{"ar" if bad else "a"})"/>')
    for s, (x, y) in POS.items():
        bad = s in alerting
        stroke = T["alert"] if bad else T["ink"]
        fill = T["alert_tint"] if bad else T["surface"]
        top = T["alert_tint"] if bad else T["grid"]
        if s == "db":
            out.append(f'<path d="M{x-40},{y-26} v52 a40,12 0 0 0 80,0 v-52" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
                       f'<ellipse cx="{x}" cy="{y-26}" rx="40" ry="12" fill="{top}" stroke="{stroke}" stroke-width="2"/>')
        else:
            out.append(f'<path d="M{x-44},{y-22} l14,-12 h88 l-14,12 z" fill="{top}" stroke="{stroke}" stroke-width="2"/>'
                       f'<path d="M{x+44},{y-22} l14,-12 v44 l-14,12 z" fill="{top}" stroke="{stroke}" stroke-width="2"/>'
                       f'<rect x="{x-44}" y="{y-22}" width="88" height="44" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
                       f'<line x1="{x-32}" y1="{y-6}" x2="{x+8}" y2="{y-6}" stroke="{stroke}" stroke-width="2"/>'
                       f'<line x1="{x-32}" y1="{y+6}" x2="{x-4}" y2="{y+6}" stroke="{stroke}" stroke-width="2"/>')
        out.append(f'<circle cx="{x+28}" cy="{y}" r="6" fill="{T["alert"] if bad else T["resolved"]}"/>')
        if last_target == s:
            out.append(f'<rect x="{x-58}" y="{y-44}" width="122" height="84" fill="none" stroke="{T["focus"]}" '
                       f'stroke-width="2.5" stroke-dasharray="4 4"/>')
        ly = y + (56 if s == "db" else 50)
        out.append(f'<text x="{x}" y="{ly}" text-anchor="middle" font-size="16" font-weight="800" '
                   f'fill="{T["alert"] if bad else T["deep"]}">{s}</text>')
        if m:
            mm = m[s]
            out.append(f'<text x="{x}" y="{ly + 18}" text-anchor="middle" font-size="13" fill="{T["muted"]}">'
                       f'err {mm["error_rate"] * 100:.0f}%  p99 {mm["p99_ms"]} ms</text>')
    out.append("</svg>")
    return "".join(out)


def steps_panel(events, upto):
    items = []
    shown = [e for e in events if e["frame"] <= upto and e["tag"] != "REASON"]
    for i, e in enumerate(shown, 1):
        items.append(f'<li><span class="num" style="background:{e["colour"]}">{i}</span>'
                     f'<span><b style="color:{e["colour"]}">{html.escape(e["label"])}</b>'
                     f'<span class="muted">{html.escape(e["text"][:120])}</span></span></li>')
    body = "".join(items) or '<li><span></span><span class="muted">Nothing yet. The agent starts at the first alert.</span></li>'
    return f'<div class="panel"><p class="ptitle">Agent steps</p><ul class="steps">{body}</ul></div>'


def status_of(run, upto):
    human = run.get("human")
    if human is not None and human.pending is not None:
        return "Waiting for you", T["pending"], T["pending_tint"]
    res = run["res"]
    if run["done"] and res is not None and upto >= len(run["frames"]) - 1:
        return ("Resolved", T["resolved"], "#E3F1EA") if res.resolved else ("Not resolved", T["alert"], T["alert_tint"])
    return "Investigating", T["ink"], T["tint"]


def overview_panel(run, upto, show_truth):
    label, col, tint = status_of(run, upto)
    events = [e for e in run["events"] if e["frame"] <= upto]
    raw = next((e["raw"] for e in reversed(events) if e["tag"] == "REASON"), "")
    diag, _, top = raw.partition(" | top: ")
    tm = re.match(r"(\S+\(\S*\)) \(([\d.]+)\) ?(.*)", top)
    plan = next((e["text"] for e in reversed(events) if e["tag"] == "CANDIDATE"), "")
    res = run["res"]
    executed = sum(1 for e in events if e["tag"] == "EXECUTE")
    unneeded = sum(1 for e in events if e["tag"] == "EXECUTE" and "no fix" in e["raw"])
    rows = [("Agent", AGENTS.get(run["arm"], run["arm"])), ("Reasoner", run["model"]),
            ("Time", f"{upto * TICK_SECONDS / 60:.1f} min (tick {upto})"),
            ("Current plan", plan.replace("Plan ", "", 1) or "none yet"), ("Actions run", f"{executed} ({unneeded} did not help)")]
    if run["done"] and res is not None and upto >= len(run["frames"]) - 1:
        rows.append(("Time to recover", f"{res.ttr_min:.1f} min" if res.ttr_min is not None else "did not recover"))
        rows.append(("Damage caused", f"{res.collateral_ticks} ticks of outage"))
    if show_truth:
        f = run["fault"]
        rows.insert(0, ("Injected fault", f"{FAULT_NAMES[f.kind]} at {f.target}"))
    table = "".join(f"<tr><td>{k}</td><td>{html.escape(str(v))}</td></tr>" for k, v in rows)
    why = ""
    if diag:
        why = f'<p style="font-size:13px;margin:8px 0 0"><b>Diagnosis.</b> {html.escape(diag[:300])}</p>'
        if tm:
            why += (f'<p style="font-size:13px;margin:6px 0 0"><b>Best fix: {tm.group(1)}</b>, confidence {tm.group(2)}. '
                    f'{html.escape(tm.group(3)[:220])}</p>')
    return (f'<div class="panel"><p class="ptitle">Agent overview</p>'
            f'<span class="badge" style="color:{col};border-color:{col};background:{tint}">{label}</span>'
            f'{why}<table class="kv" style="margin-top:10px">{table}</table></div>')


ACTION_COPY = {
    "drain_node": ("Drain {t}?", "Every pod on {t} moves to another node and {t} stops taking work.",
                   "Drain {t}", "Don't drain"),
    "restart_database": ("Restart the database?", "The database goes offline for about 1.5 minutes. Cart, catalog "
                         "and inventory fail until it is back.", "Restart database", "Keep it running"),
}


def log_table(events, upto):
    legend = "".join(f'<span style="margin-right:14px;font-size:13px"><span class="dot" style="background:{c}"></span>'
                     f'{l}</span>' for l, c in LEGEND.items())
    rows = []
    for i, e in enumerate([e for e in events if e["frame"] <= upto], 1):
        now = ' class="now"' if e["frame"] == upto else ""
        rows.append(f'<tr{now}><td>{i}</td><td>{e["frame"] * TICK_SECONDS / 60:.1f} min</td>'
                    f'<td><span class="dot" style="background:{e["colour"]}"></span>{e["label"]}</td>'
                    f'<td>{html.escape(e["text"][:180])}</td></tr>')
    body = "".join(rows) or '<tr><td colspan="4" class="muted">No events yet.</td></tr>'
    return (f'<div class="panel"><p class="ptitle" style="display:inline-block;margin-right:18px">Decision log</p>{legend}'
            f'<table class="log"><tr><th style="width:44px">No.</th><th style="width:90px">Time</th>'
            f'<th style="width:130px">Event</th><th>Detail</th></tr>{body}</table></div>')


# ---------------------------------------------------------------- page
css()
ss = st.session_state
ss.setdefault("run", None)
ss.setdefault("cursor", 0)
ss.setdefault("playing", False)
ss.setdefault("compare", None)

st.markdown('<p class="apptitle">Reconsider before you act</p>'
            '<p class="appsub">A self-healing agent for microservices, step by step</p>', unsafe_allow_html=True)

folders = sorted(d for d in glob.glob(os.path.join("results", "llm_*"))
                 if os.path.exists(os.path.join(d, "llm_cache.jsonl")))
_q = (st.query_params.get("load") or "").split(",")
_qf = os.path.join("results", _q[0]) if len(_q) == 4 else None
c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.3, 0.7, 0.9, 1.6, 1.0], vertical_alignment="bottom")
mode = c1.selectbox("Mode", ["Recorded run", "Live run"])
if mode == "Recorded run":
    folder = c2.selectbox("Experiment", folders or ["(no results/llm_* folder)"],
                          index=folders.index(_qf) if _qf in folders else 0)
    episode = c3.number_input("Episode", 0, 99, int(_q[1]) if len(_q) == 4 else 0)
    index = c4.number_input("Incident", 0, 15, int(_q[2]) if len(_q) == 4 else 5)
    arm = c5.selectbox("Agent", list(AGENTS), format_func=lambda a: AGENTS[a],
                       index=list(AGENTS).index(_q[3]) if len(_q) == 4 and _q[3] in AGENTS else 0)
    if c6.button("Load incident", type="primary", use_container_width=True) and folders:
        noise = 0.5 if "noisy" in folder else 0.0
        with st.spinner("Replaying the recorded run"):
            ss.run = replay(folder, int(episode), int(index), arm, noise)
            ss.compare = replay(folder, int(episode), int(index), "B1-ReAct", noise) if arm != "B1-ReAct" else None
        ss.cursor, ss.playing = 0, False
else:
    kind = c2.selectbox("Fault to inject", FAULT_KINDS, format_func=lambda k: FAULT_NAMES[k])
    targets = {"slow_query": ["db"], "node_pressure": ["node-a", "node-b"],
               "shifting": ["payment", "inventory"]}.get(
        kind, ["frontend", "cart", "catalog", "checkout", "payment", "inventory"])
    target = c3.selectbox("Where", targets)
    arm = c4.selectbox("Agent", list(AGENTS), format_func=lambda a: AGENTS[a].split(" (")[0])
    reasoner = c5.selectbox("Reasoner", ["Scripted (instant)", "LLM (about 40 s per call)"])
    if c6.button("Start incident", type="primary", use_container_width=True):
        ss.run = start_live(kind, target, arm, reasoner.startswith("LLM"), 0.0)
        ss.compare, ss.cursor, ss.playing = None, 0, True

# A link such as ?load=llm_noisy,0,11,Ours opens straight onto a recorded incident (handy for the talk).
q = st.query_params.get("load")
if q and ss.run is None:
    try:
        f, e, i, a = q.split(",")
        path = os.path.join("results", f)
        ss.run = replay(path, int(e), int(i), a, 0.5 if "noisy" in f else 0.0)
        ss.compare = replay(path, int(e), int(i), "B1-ReAct", 0.5 if "noisy" in f else 0.0) if a != "B1-ReAct" else None
        ss.cursor = int(st.query_params.get("at", 0))
    except Exception as err:
        st.error(f"Could not open {q}: {err}")

run = ss.run
if run is None:
    st.markdown('<div class="panel"><p class="ptitle">No incident loaded</p><p>Pick a recorded run and press '
                '<b>Load incident</b>, or switch to Live run to inject a fault yourself.</p></div>',
                unsafe_allow_html=True)
    st.stop()

if run.get("error"):
    st.error(f"The run stopped: {run['error']}")

live = hasattr(run.get("human"), "answered")   # isinstance would break: the class is redefined on every rerun
if live and not run["done"] and hasattr(run["agent"], "res"):
    run["events"] = parse_events(run["agent"].res.trace)
n = len(run["frames"])
last = max(0, n - 1)
if live and not run["done"]:
    ss.cursor = last
ss.cursor = min(ss.cursor, last)
upto = ss.cursor
obs = run["frames"][upto] if n else None

left, mid, right = st.columns([1.05, 2.45, 1.25])
with left:
    st.markdown(steps_panel(run["events"], upto), unsafe_allow_html=True)
with mid:
    MID = st.container()
with right:
    human = run.get("human")
    if live and human.pending is not None:
        a = human.pending
        title, body, yes, no = ACTION_COPY.get(a.name, ("Run {a}?", "This is a high-risk action.", "Run it", "Don't run it"))
        fmt = dict(t=a.target, a=str(a))
        st.markdown(f'<div class="approval"><p class="ptitle">Approval needed</p><p><b>{title.format(**fmt)}</b></p>'
                    f'<p>{body.format(**fmt)}</p></div>', unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        if b1.button(yes.format(**fmt), type="primary", use_container_width=True):
            human.decision, human.pending = True, None
            human.answered.set()
            time.sleep(0.3)
            st.rerun()
        if b2.button(no.format(**fmt), use_container_width=True):
            human.decision, human.pending = False, None
            human.answered.set()
            time.sleep(0.3)
            st.rerun()
    show_truth = st.toggle("Show the injected fault", value=False)
    st.markdown(overview_panel(run, upto, show_truth), unsafe_allow_html=True)
    cmp = ss.compare
    if cmp is not None and cmp["res"] is not None and upto >= last:
        a, b = run["res"], cmp["res"]
        rows = [("Actions run", a.executed, b.executed), ("Did not help", a.unnecessary, b.unnecessary),
                ("Damage (ticks)", a.collateral_ticks, b.collateral_ticks),
                ("Recovered in", f"{a.ttr_min:.1f} min" if a.ttr_min is not None else "no",
                 f"{b.ttr_min:.1f} min" if b.ttr_min is not None else "no")]
        tbl = "".join(f"<tr><td>{k}</td><td>{x}</td><td>{y}</td></tr>" for k, x, y in rows)
        st.markdown(f'<div class="panel"><p class="ptitle">Same incident, B1</p><table class="kv">'
                    f'<tr><td></td><td><b>This agent</b></td><td><b>B1 ReAct</b></td></tr>{tbl}</table></div>',
                    unsafe_allow_html=True)

with MID:
    acts = [e["raw"].split(" ")[0] for e in run["events"] if e["tag"] == "EXECUTE" and e["frame"] <= upto]
    drained = {a.split("(")[1].rstrip(")") for a in acts if a.startswith("drain_node")}
    last_target = acts[-1].split("(")[1].rstrip(")") if acts else None
    st.markdown(f'<div class="panel"><p class="ptitle">Service map</p>{service_map(obs, last_target, drained)}</div>',
                unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns([1, 1, 1, 1, 2.6], vertical_alignment="center")
    if p1.button("Play", type="primary", use_container_width=True):
        ss.playing = True
    if p2.button("Pause", use_container_width=True):
        ss.playing = False
    if p3.button("Step", use_container_width=True):
        ss.playing = False
        ss.cursor = min(last, ss.cursor + 1)
        st.rerun()
    if p4.button("Reset", use_container_width=True):
        ss.playing, ss.cursor = False, 0
        st.rerun()
    note = run["source"]
    if run.get("misses"):
        note += f" ({run['misses']} replies were not in the recording and came from the scripted reasoner)"
    if live and not run["done"] and str(run["model"]) != "scripted reasoner" and (human.pending is None):
        note += f". Asking {run['model']}; one answer takes about 40 seconds"
    p5.markdown(f'<p class="muted" style="margin:0;font-size:13px">{html.escape(note)}</p>', unsafe_allow_html=True)

    st.markdown(log_table(run["events"], upto), unsafe_allow_html=True)


# keep the page moving
if live and not run["done"]:
    if human.pending is None:
        time.sleep(0.6)
        st.rerun()
elif ss.playing and ss.cursor < last:
    time.sleep(0.8)
    ss.cursor += 1
    st.rerun()
elif ss.playing:
    ss.playing = False
