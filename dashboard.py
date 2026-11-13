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


# ---------------------------------------------------------------- results (charts made from results/*)
import altair as alt
import pandas as pd

ARM_ORDER = ["B1-ReAct", "B2-ActRollback", "Ours-noCompile", "Ours"]
ARM_LABEL = {"B1-ReAct": "B1 ReAct", "B2-ActRollback": "B2 act + roll back",
             "Ours-noCompile": "Ours, no memory", "Ours": "Ours"}
ARM_COLOUR = {"B1 ReAct": "#9DB1D9", "B2 act + roll back": "#6982B8", "Ours, no memory": "#1F5FE0", "Ours": "#0032A0"}
SETTING = {"llm_clear": "Clear evidence", "llm_noisy": "Noisy evidence"}


def alt_theme(chart, height=260):
    if not isinstance(chart, alt.FacetChart):
        chart = chart.properties(height=height)
    return (chart
            .configure(font="Iosevka, ui-monospace, Menlo, Consolas, monospace", background=T["surface"])
            .configure_axis(labelLimit=260, labelColor=T["muted"], titleColor=T["ink"], gridColor=T["grid"], domainColor=T["line"],
                            tickColor=T["line"], labelFontSize=13, titleFontSize=13, titleFontWeight=700)
            .configure_legend(labelColor=T["ink"], titleColor=T["ink"], labelFontSize=13, orient="bottom", title=None)
            .configure_view(stroke=None)
            .configure_header(labelColor=T["ink"], labelFontSize=14, labelFontWeight=800, title=None))


@st.cache_data
def load_incidents(folder):
    df = pd.read_csv(os.path.join(folder, "incidents.csv"))
    df["agent_label"] = df["agent"].map(ARM_LABEL).fillna(df["agent"])
    df["setting"] = SETTING.get(os.path.basename(folder), os.path.basename(folder))
    return df


def bootstrap(values, n=1000, seed=0):
    rng = random.Random(seed)
    vals = list(values)
    means = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return means[int(.025 * n)], means[int(.975 * n)]


def mean_ci(df, col):
    rows = []
    for (setting, arm), g in df.groupby(["setting", "agent_label"]):
        v = g[col].dropna().astype(float)
        lo, hi = bootstrap(v)
        rows.append({"setting": setting, "agent": arm, "mean": v.mean(), "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def bar_ci(stats, title):
    order = [ARM_LABEL[a] for a in ARM_ORDER]
    base = alt.Chart(stats).encode(
        y=alt.Y("agent:N", sort=order, title=None),
        color=alt.Color("agent:N", scale=alt.Scale(domain=list(ARM_COLOUR), range=list(ARM_COLOUR.values())), legend=None))
    bars = base.mark_bar(height=22).encode(x=alt.X("mean:Q", title=title),
                                           tooltip=["agent", alt.Tooltip("mean:Q", format=".2f"),
                                                    alt.Tooltip("lo:Q", format=".2f"), alt.Tooltip("hi:Q", format=".2f")])
    err = base.mark_rule(color=T["deep"], strokeWidth=2).encode(x="lo:Q", x2="hi:Q")
    labels = base.mark_text(align="left", dx=6, color=T["deep"], fontSize=13, fontWeight=700).encode(
        x="hi:Q", text=alt.Text("mean:Q", format=".2f"))
    return alt.layer(bars, err, labels, data=stats).properties(height=150, width=300).facet(
        column=alt.Column("setting:N", title=None))


def headline(df):
    out = []
    for setting, g in df.groupby("setting"):
        b, o = g[g.agent == "B1-ReAct"], g[g.agent == "Ours"]
        out.append({"setting": setting,
                    "unneeded": (b.unnecessary.mean(), o.unnecessary.mean()),
                    "damage": (int(b.collateral_ticks.sum()), int(o.collateral_ticks.sum())),
                    "mttr": (b.ttr_min.mean(), o.ttr_min.mean()),
                    "calls": (b.llm_calls.mean(), o.llm_calls.mean())})
    return out


def figure_card(label, before, after, fmt, note):
    return (f'<div class="panel" style="min-height:120px"><p class="label">{label}</p>'
            f'<p class="figure">{fmt.format(before)} <span class="arrow">to</span> {fmt.format(after)}</p>'
            f'<p class="muted" style="font-size:13px;margin:0">{note}</p></div>')


def results_tab():
    folders = [d for d in ("results/llm_clear", "results/llm_noisy") if os.path.exists(os.path.join(d, "incidents.csv"))]
    if not folders:
        st.markdown('<div class="panel"><p class="ptitle">No LLM results yet</p><p>Run run_experiments.py with '
                    '--reasoner llm and the results appear here.</p></div>', unsafe_allow_html=True)
        return
    df = pd.concat([load_incidents(f) for f in folders])
    n_ep = df.episode.nunique()
    st.markdown(f'<p class="lead">B1 against our agent, muse-glimmer-30b on the same {n_ep * 16} incidents per setting '
                f'({n_ep} episodes of 16). Bars show the mean, lines the 95% bootstrap interval.</p>',
                unsafe_allow_html=True)
    for h in headline(df):
        st.markdown(f'<p class="ptitle" style="margin-top:6px">{h["setting"]}</p>', unsafe_allow_html=True)
        c = st.columns(4)
        b, o = h["unneeded"]
        c[0].markdown(figure_card("Actions that did not help", b, o, "{:.2f}",
                                  f"per incident, {100 * (o - b) / b:+.0f}%"), unsafe_allow_html=True)
        b, o = h["damage"]
        c[1].markdown(figure_card("Outage the agent caused", b, o, "{}", "ticks in total (1 tick = 30 s)"),
                      unsafe_allow_html=True)
        b, o = h["mttr"]
        c[2].markdown(figure_card("Time to recover", b, o, "{:.1f}", f"minutes, {o - b:+.1f} min: the price of checking"),
                      unsafe_allow_html=True)
        b, o = h["calls"]
        c[3].markdown(figure_card("LLM calls", b, o, "{:.2f}", f"per incident, {100 * (o - b) / b:+.0f}% thanks to memory"),
                      unsafe_allow_html=True)

    st.markdown('<p class="ptitle" style="margin-top:14px">Actions that did not help, per incident</p>', unsafe_allow_html=True)
    st.altair_chart(alt_theme(bar_ci(mean_ci(df, "unnecessary"), "per incident"), 150), use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.markdown('<p class="ptitle">Where they happened</p>', unsafe_allow_html=True)
        sub = df[df.agent.isin(["B1-ReAct", "Ours"])]
        by = sub.groupby(["setting", "agent_label", "fault"], as_index=False)["unnecessary"].sum()
        by = by[by.groupby(["setting", "fault"]).unnecessary.transform("sum") > 0]
        by["fault"] = by["fault"].map(FAULT_NAMES)
        ch = alt.Chart(by).mark_bar().encode(
            y=alt.Y("fault:N", title=None), x=alt.X("unnecessary:Q", title="actions in total"),
            yOffset="agent_label:N",
            color=alt.Color("agent_label:N", scale=alt.Scale(domain=["B1 ReAct", "Ours"],
                                                             range=[ARM_COLOUR["B1 ReAct"], ARM_COLOUR["Ours"]])),
            tooltip=["setting", "fault", "agent_label", "unnecessary"]).properties(height=110, width=520).facet(
            row=alt.Row("setting:N", title=None))
        st.altair_chart(alt_theme(ch, 110), use_container_width=True)
    with r:
        st.markdown('<p class="ptitle">Time to recover</p>', unsafe_allow_html=True)
        st.altair_chart(alt_theme(bar_ci(mean_ci(df, "ttr_min"), "minutes"), 150), use_container_width=True)
        st.markdown('<p class="ptitle">Memory: LLM calls on a repeated fault</p>', unsafe_allow_html=True)
        rep = df[df.agent.isin(["Ours-noCompile", "Ours"])].copy()
        rep["seen"] = rep.occurrence.map(lambda o: "first time" if o == 1 else "seen before")
        rep = rep.groupby(["setting", "agent_label", "seen"], as_index=False)["llm_calls"].mean()
        ch = alt.Chart(rep).mark_bar().encode(
            x=alt.X("seen:N", title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("llm_calls:Q", title="calls per incident"), xOffset="agent_label:N",
            color=alt.Color("agent_label:N", scale=alt.Scale(domain=["Ours, no memory", "Ours"],
                                                             range=[ARM_COLOUR["Ours, no memory"], ARM_COLOUR["Ours"]])),
            tooltip=["setting", "seen", "agent_label", alt.Tooltip("llm_calls:Q", format=".2f")]
        ).properties(height=170, width=240).facet(column=alt.Column("setting:N", title=None))
        st.altair_chart(alt_theme(ch, 170), use_container_width=True)

    with st.expander("Scripted reasoner: ablation and sensitivity (30 episodes each)"):
        ab = "results/ablation/summary.json"
        if os.path.exists(ab):
            import json
            s = json.load(open(ab))["summary"]
            rows = [{"Variant": k, "Did not help / incident": round(v["unnecessary_per_incident"], 2),
                     "Outage ticks": v["collateral_ticks_total"], "Recover (min)": round(v["mttr_min"], 2)}
                    for k, v in s.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        pts = []
        for slip in ("0.0", "0.25", "0.5"):
            p = f"results/slip_{slip}/summary.json"
            if os.path.exists(p):
                import json
                for k, v in json.load(open(p))["summary"].items():
                    pts.append({"slip": float(slip), "agent": ARM_LABEL.get(k, k), "did not help": v["unnecessary_per_incident"]})
        if pts:
            ch = alt.Chart(pd.DataFrame(pts)).mark_line(point=True).encode(
                x=alt.X("slip:Q", title="chance the reasoner jumps to a generic fix"),
                y=alt.Y("did not help:Q", title="did not help / incident"),
                color=alt.Color("agent:N", scale=alt.Scale(domain=list(ARM_COLOUR), range=list(ARM_COLOUR.values()))))
            st.altair_chart(alt_theme(ch, 240), use_container_width=True)
    st.markdown('<p class="muted" style="font-size:13px">Every chart has a menu (three dots, top right) to save it '
                'as PNG or SVG for the slides and the paper.</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------- replay: two agents side by side
@st.cache_data
def find_stories(folder):
    """Pick incidents from the recorded run that show each mechanism, so nobody has to hunt for them."""
    import json
    traces = {}
    path = os.path.join(folder, "traces.jsonl")
    if not os.path.exists(path):
        return []
    for line in open(path, encoding="utf-8"):
        t = json.loads(line)
        traces[(t["episode"], t["index"], t["agent"])] = t
    rows = pd.read_csv(os.path.join(folder, "incidents.csv"))
    ri = {(r.episode, r["index"], r.agent): r for _, r in rows.iterrows()}
    stories = []

    def first(pred, title, why):
        for (e, i, a), t in sorted(traces.items()):
            if a == "Ours" and pred(e, i, t):
                stories.append({"title": title, "why": why, "episode": e, "index": i, "fault": t["fault"],
                                "target": t["target"]})
                return

    b1 = lambda e, i: ri.get((e, i, "B1-ReAct"))
    first(lambda e, i, t: b1(e, i) is not None and b1(e, i).collateral_ticks > 0,
          "B1 restarts the database for a blip",
          "A short blip that would heal on its own. B1 keeps acting and ends with a database restart.")
    first(lambda e, i, t: t["fault"] == "transient" and b1(e, i) is not None and b1(e, i).unnecessary > ri[(e, i, "Ours")].unnecessary,
          "A blip that heals itself", "Both agents see the same alert. Ours stops sooner.")
    first(lambda e, i, t: t["fault"] == "node_pressure" and any("ESCALATE" in l for l in t["trace"]),
          "Node out of disk: ask a person first", "Draining a node is high risk, so ours waits for approval.")
    first(lambda e, i, t: any("RECONSIDER" in l and "changed" in l for l in t["trace"]),
          "The symptom moves before acting", "Ours re-checks, sees a different alert and plans again.")
    first(lambda e, i, t: any("procedure store" in l for l in t["trace"]),
          "A fault it has seen before", "Ours reuses the fix it saved last time instead of asking the LLM.")
    return stories


def compact_map(obs, events, upto):
    acts = [e["raw"].split(" ")[0] for e in events if e["tag"] == "EXECUTE" and e["frame"] <= upto]
    drained = {a.split("(")[1].rstrip(")") for a in acts if a.startswith("drain_node")}
    target = acts[-1].split("(")[1].rstrip(")") if acts else None
    return service_map(obs, target, drained)


def side(run, upto, title):
    n = len(run["frames"])
    obs = run["frames"][min(upto, n - 1)] if n else None
    events = [e for e in run["events"] if e["frame"] <= upto and e["tag"] != "REASON"]
    res = run["res"]
    finished = upto >= n - 1
    if finished:
        badge = ("Recovered" if res.resolved else "Not recovered", T["resolved"] if res.resolved else T["alert"])
    elif obs and obs["alerts"]:
        badge = ("Working on it", T["ink"])
    elif events:
        badge = ("No alerts now, watching", T["resolved"])
    else:
        badge = ("Waiting for an alert", T["muted"])
    items = "".join(f'<li><span class="num" style="background:{e["colour"]}">{k}</span><span>'
                    f'<b style="color:{e["colour"]}">{e["label"]}</b> {html.escape(e["text"])}</span></li>'
                    for k, e in enumerate(events, 1)) or '<li><span></span><span class="muted">Nothing yet.</span></li>'
    done = [e for e in run["events"] if e["tag"] == "EXECUTE"]
    score = ""
    if finished:
        score = (f'<table class="kv" style="margin-top:8px"><tr><td>Actions</td><td>{res.executed} '
                 f'({res.unnecessary} did not help)</td></tr><tr><td>Outage caused</td><td>{res.collateral_ticks} ticks'
                 f'</td></tr><tr><td>Recovered in</td><td>{"%.1f min" % res.ttr_min if res.ttr_min is not None else "no"}'
                 f'</td></tr></table>')
    st.markdown(f'<div class="panel"><p class="ptitle" style="margin-bottom:4px">{title}</p>'
                f'<span class="badge" style="color:{badge[1]};border-color:{badge[1]}">{badge[0]}</span>'
                f'{compact_map(obs, run["events"], upto)}<ul class="steps big">{items}</ul>{score}</div>',
                unsafe_allow_html=True)


def replay_tab():
    folders = [d for d in ("results/llm_noisy", "results/llm_clear") if os.path.exists(os.path.join(d, "llm_cache.jsonl"))]
    if not folders:
        st.markdown('<div class="panel"><p class="ptitle">No recorded runs</p><p>Run the LLM experiments first.</p></div>',
                    unsafe_allow_html=True)
        return
    stories = []
    for f in folders:
        for s in find_stories(f):
            s["folder"] = f
            if s["title"] not in [x["title"] for x in stories]:
                stories.append(s)
    pick = st.radio("Incident", list(range(len(stories))), horizontal=True, label_visibility="collapsed",
                    format_func=lambda k: stories[k]["title"])
    s = stories[pick]
    key = (s["folder"], s["episode"], s["index"])
    if ss.get("pair_key") != key:
        noise = 0.5 if "noisy" in s["folder"] else 0.0
        ss.pair = (replay(s["folder"], s["episode"], s["index"], "B1-ReAct", noise),
                   replay(s["folder"], s["episode"], s["index"], "Ours", noise))
        ss.pair_key, ss.t = key, 0
    a, b = ss.pair
    last = max(len(a["frames"]), len(b["frames"])) - 1
    st.markdown(f'<p class="lead">{html.escape(s["why"])} <span class="muted">Hidden from both agents: '
                f'{FAULT_NAMES[s["fault"]].lower()} at {s["target"]}. Recorded with muse-glimmer-30b, '
                f'{SETTING[os.path.basename(s["folder"])].lower()}.</span></p>', unsafe_allow_html=True)
    c = st.columns([1, 1, 1, 1, 4], vertical_alignment="center")
    if c[0].button("Back", use_container_width=True):
        ss.t = max(0, ss.t - 1)
    if c[1].button("Next step", type="primary", use_container_width=True):
        ss.t = min(last, ss.t + 1)
    if c[2].button("Show the end", use_container_width=True):
        ss.t = last
    if c[3].button("Start over", use_container_width=True):
        ss.t = 0
    c[4].markdown(f'<p class="clock">{ss.t * TICK_SECONDS / 60:.1f} min after the fault</p>', unsafe_allow_html=True)
    l, r = st.columns(2)
    with l:
        side(a, ss.t, "B1: act straight away")
    with r:
        side(b, ss.t, "Ours: check, then act")


# ---------------------------------------------------------------- live
def live_tab():
    c = st.columns([1.6, 1, 1.4, 1], vertical_alignment="bottom")
    kind = c[0].selectbox("Fault to inject", FAULT_KINDS, format_func=lambda k: FAULT_NAMES[k],
                          index=FAULT_KINDS.index("node_pressure"))
    targets = {"slow_query": ["db"], "node_pressure": ["node-a", "node-b"], "shifting": ["payment", "inventory"]}.get(
        kind, ["frontend", "cart", "catalog", "checkout", "payment", "inventory"])
    target = c[1].selectbox("Where", targets)
    reasoner = c[2].selectbox("Reasoner", ["Scripted (instant)", "muse-glimmer-30b (about 40 s per call)"])
    if c[3].button("Inject fault", type="primary", use_container_width=True):
        ss.live = start_live(kind, target, "Ours", reasoner.startswith("muse"), 0.0)
    run = ss.get("live")
    if run is None:
        st.markdown('<p class="lead">Inject a fault and watch our agent handle it. When it wants to do something '
                    'risky, it stops and asks you.</p>', unsafe_allow_html=True)
        return
    if hasattr(run.get("agent"), "res"):
        run["events"] = parse_events(run["agent"].res.trace)
    human = run["human"]
    l, r = st.columns([1.6, 1])
    with l:
        n = len(run["frames"])
        obs = run["frames"][-1] if n else None
        st.markdown(f'<div class="panel">{compact_map(obs, run["events"], n)}</div>', unsafe_allow_html=True)
    with r:
        if human.pending is not None:
            a = human.pending
            title, body, yes, no = ACTION_COPY.get(a.name, ("Run {a}?", "This is a high-risk action.", "Run it", "Don't run it"))
            fmt = dict(t=a.target, a=str(a))
            st.markdown(f'<div class="approval"><p class="ptitle">Your call</p><p><b>{title.format(**fmt)}</b></p>'
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
        events = [e for e in run["events"] if e["tag"] != "REASON"]
        items = "".join(f'<li><span class="num" style="background:{e["colour"]}">{k}</span><span>'
                        f'<b style="color:{e["colour"]}">{e["label"]}</b> {html.escape(e["text"])}</span></li>'
                        for k, e in enumerate(events, 1))
        status = "Done" if run["done"] else ("Waiting for you" if human.pending is not None else "Working")
        if not run["done"] and human.pending is None and run["model"] != "scripted reasoner":
            status += f", asking {run['model']}"
        st.markdown(f'<div class="panel"><p class="ptitle">{status}</p><ul class="steps big">{items}</ul></div>',
                    unsafe_allow_html=True)
    if run.get("error"):
        st.error(f"The run stopped: {run['error']}")
    if not run["done"] and human.pending is None:
        time.sleep(0.6)
        st.rerun()


# ---------------------------------------------------------------- page
css()
st.markdown(f"""<style>
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1.5px solid {T['ink']}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 0; padding: 8px 22px; }}
.stTabs [data-baseweb="tab"] p, .stTabs button[role="tab"] p {{ font-size: 18px !important; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }}
.stTabs [aria-selected="true"] {{ background: {T['ink']}; }}
.stTabs [aria-selected="true"] p {{ color: {T['surface']} !important; }}
p.lead {{ font-size: 17px !important; line-height: 1.5; max-width: 70em; margin: 6px 0 12px; }}
.label {{ font-size: 13px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; margin: 0 0 6px; }}
.panel p.figure {{ font-size: 34px !important; line-height: 1.15 !important; font-weight: 800; color: {T['deep']} !important; margin: 0 0 4px; letter-spacing: .02em; }}
.panel p.figure .arrow {{ font-size: 15px !important; font-weight: 400; color: {T['muted']} !important; }}
p.clock {{ font-size: 20px !important; font-weight: 800; color: {T['deep']} !important; margin: 0; }}
.steps.big li {{ font-size: 15px; }}
.steps.big b {{ display: inline; margin-right: 6px; }}
div[role="radiogroup"] label p {{ font-size: 15px; }}
span[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
button[role="tab"] p, button[data-baseweb="tab"] p, button[role="tab"] div {{ font-size: 18px !important;
  font-weight: 800 !important; letter-spacing: .06em; text-transform: uppercase; }}
</style>""", unsafe_allow_html=True)
ss = st.session_state
st.markdown('<p class="apptitle">Reconsider before you act</p>'
            '<p class="appsub">Prapatsorn Sangrod, CPE 633 term project</p>', unsafe_allow_html=True)
t1, t2, t3 = st.tabs(["Replay", "Results", "Live"])
with t1:
    replay_tab()
with t2:
    results_tab()
with t3:
    live_tab()
