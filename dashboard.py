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
from rbya.sim import DEPS, FAULT_KINDS, NODES, TICK_SECONDS, Fault, Simulator, sample_fault

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



TH_FAULTS = {"bad_deploy": "deploy เวอร์ชันเสีย", "config_error": "config ผิด", "memory_leak": "หน่วยความจำรั่ว",
             "traffic_spike": "traffic พุ่ง", "transient": "สะดุดชั่วคราว (หายเอง)", "slow_query": "query ค้างในฐานข้อมูล",
             "shifting": "อาการย้ายไปที่ต้นเหตุจริง", "node_pressure": "เครื่อง (node) ดิสก์เต็ม"}
TXT = {
    "en": {"subtitle": "Prapatsorn Sangrod, CPE 633 term project", "views": ["Replay", "Results", "Live"],
           "animation": "Animation", "animation_help": "Moving packets and pulsing alerts on the map",
           "back": "Back", "next": "Next step", "play": "Play", "pause": "Pause", "start_over": "Start over",
           "clock": "{m:.1f} min after the fault", "b1_title": "B1: acts straight away", "ours_title": "Ours: checks, then acts",
           "working": "Working on it", "waiting_alert": "Waiting for an alert", "watching": "No alerts, watching",
           "recovered": "Recovered", "not_recovered": "Not recovered", "details": "Step details",
           "nothing_yet": "Nothing yet. Both agents start at the first alert.", "recovered_in": "Recovered in {m:.1f} min",
           "did_not_recover": "Did not recover", "actions": "Actions", "actions_val": "{n}, {u} did not help",
           "outage": "Outage caused", "outage_val": "{m} min ({t} ticks)", "diagnosis": "LLM diagnosis.",
           "hidden": "Hidden from both agents: {f} at {t}.", "log": "Decision log", "newest": "newest first",
           "cols": ["No.", "Time", "Agent", "Event", "What happened"], "press_next": "Press Next step to start.",
           "min": "{m:.1f} min", "ours": "Ours",
           "no_runs_t": "No recorded runs yet", "no_runs": "The replay needs the LLM experiment results. Run run_experiments.py with --reasoner llm, then reload this page.",
           "fault_inject": "Fault to inject", "where": "Where", "reasoner": "Reasoner",
           "reasoners": ["Scripted (instant)", "muse-glimmer-30b (about 40 s per call)"], "inject": "Inject fault",
           "no_live_t": "No incident running",
           "no_live": "Pick a fault and press Inject fault. Our agent handles it on its own, and stops to ask you before anything risky, such as draining a node or restarting the database.",
           "your_call": "Your call", "done": "Done", "waiting_you": "Waiting for you", "working_live": "Working",
           "asking": ", asking {m}", "stopped": "The run stopped: {e}",
           "agent_acts": "AGENT ACTS", "asking_oncall": "ASKING ON-CALL",
           "approve": {"drain_node": ("Drain {t}?", "Every pod on {t} moves to another node and {t} stops taking work.", "Drain {t}", "Don't drain"),
                       "restart_database": ("Restart the database?", "The database goes offline for about 1.5 minutes. Cart, catalog and inventory fail until it is back.", "Restart database", "Keep it running"),
                       "_": ("Run {a}?", "This is a high-risk action.", "Run it", "Don't run it")},
           "ev": {"Alert": "Alert", "Plan": "Plan", "Reasoning": "Reasoning", "Hold": "Hold", "Re-check": "Re-check",
                  "Human": "Human", "Action": "Action", "Undo": "Undo", "Post-check": "Post-check", "Memory": "Memory"},
           "stories": {}},
    "th": {"subtitle": "Prapatsorn Sangrod, term project วิชา CPE 633", "views": ["เล่นซ้ำ", "ผลการทดลอง", "ทดลองสด"],
           "animation": "ภาพเคลื่อนไหว", "animation_help": "จุด request วิ่งตามเส้นและวงกระเพื่อมรอบ service ที่เสีย",
           "back": "ย้อนกลับ", "next": "ขั้นถัดไป", "play": "เล่นต่อเนื่อง", "pause": "หยุด", "start_over": "เริ่มใหม่",
           "clock": "{m:.1f} นาทีหลังเกิดปัญหา", "b1_title": "B1: ลงมือทันที", "ours_title": "ของเรา: ตรวจซ้ำก่อนลงมือ",
           "working": "กำลังแก้", "waiting_alert": "รอการแจ้งเตือน", "watching": "ไม่มีแจ้งเตือน เฝ้าดูอยู่",
           "recovered": "กู้คืนแล้ว", "not_recovered": "ยังไม่กู้คืน", "details": "รายละเอียดขั้นนี้",
           "nothing_yet": "ยังไม่มีอะไร ทั้งสองฝั่งจะเริ่มทำงานเมื่อมีการแจ้งเตือนครั้งแรก", "recovered_in": "กู้คืนใน {m:.1f} นาที",
           "did_not_recover": "กู้คืนไม่ได้", "actions": "action ที่ทำ", "actions_val": "{n} ครั้ง ไม่ช่วย {u} ครั้ง",
           "outage": "ระบบล่มเพราะ agent", "outage_val": "{m} นาที ({t} tick)", "diagnosis": "LLM วินิจฉัยว่า",
           "hidden": "ปัญหาที่ใส่ไว้ (ทั้งสองฝั่งไม่เห็น): {f} ที่ {t}", "log": "บันทึกการตัดสินใจ", "newest": "ล่าสุดอยู่บน",
           "cols": ["ลำดับ", "เวลา", "ฝั่ง", "เหตุการณ์", "เกิดอะไรขึ้น"], "press_next": "กด ขั้นถัดไป เพื่อเริ่ม",
           "min": "{m:.1f} นาที", "ours": "ของเรา",
           "no_runs_t": "ยังไม่มีผลที่บันทึกไว้", "no_runs": "หน้าเล่นซ้ำต้องใช้ผลการทดลองกับ LLM รัน run_experiments.py --reasoner llm ก่อน แล้วโหลดหน้านี้ใหม่",
           "fault_inject": "ปัญหาที่จะใส่", "where": "ที่ไหน", "reasoner": "ตัวคิด",
           "reasoners": ["สคริปต์ (ทันที)", "muse-glimmer-30b (ราว 40 วินาทีต่อครั้ง)"], "inject": "ใส่ปัญหา",
           "no_live_t": "ยังไม่มีเหตุการณ์",
           "no_live": "เลือกปัญหาแล้วกด ใส่ปัญหา agent ของเราจะจัดการเอง และจะหยุดถามคุณก่อนทำอะไรเสี่ยง เช่น ย้ายงานออกจากเครื่องหรือรีสตาร์ตฐานข้อมูล",
           "your_call": "รอคุณตัดสินใจ", "done": "เสร็จแล้ว", "waiting_you": "รอคุณตัดสินใจ", "working_live": "กำลังทำงาน",
           "asking": " กำลังถาม {m}", "stopped": "การรันหยุดกลางคัน: {e}",
           "agent_acts": "AGENT ลงมือ", "asking_oncall": "รอคนอนุมัติ",
           "approve": {"drain_node": ("ย้ายงานทั้งหมดออกจาก {t}?", "ทุก pod บน {t} จะย้ายไปเครื่องอื่น และ {t} จะหยุดรับงาน", "ย้ายงานออกจาก {t}", "ยังไม่ต้องย้าย"),
                       "restart_database": ("รีสตาร์ตฐานข้อมูล?", "ฐานข้อมูลจะปิดไปราว 1.5 นาที cart, catalog และ inventory จะใช้งานไม่ได้จนกว่าจะกลับมา", "รีสตาร์ตฐานข้อมูล", "ปล่อยไว้ก่อน"),
                       "_": ("ทำ {a}?", "action นี้มีความเสี่ยงสูง", "ทำเลย", "ยังไม่ทำ")},
           "ev": {"Alert": "แจ้งเตือน", "Plan": "วางแผน", "Reasoning": "วิเคราะห์", "Hold": "รอข้อมูล", "Re-check": "ตรวจซ้ำ",
                  "Human": "คน", "Action": "ลงมือ", "Undo": "ย้อนกลับ", "Post-check": "ตรวจหลังทำ", "Memory": "ความจำ"},
           "stories": {"B1 restarts the database for a blip": ("B1 รีสตาร์ตฐานข้อมูลเพราะอาการชั่วคราว", "อาการสะดุดสั้น ๆ ที่หายเองได้ B1 ลงมือไม่หยุดจนจบที่การรีสตาร์ตฐานข้อมูล"),
                       "A blip that heals itself": ("อาการชั่วคราวที่หายเอง", "ทั้งสองฝั่งเห็นแจ้งเตือนเดียวกัน ของเราหยุดได้เร็วกว่า"),
                       "Node out of disk: ask a person first": ("ดิสก์เต็ม: ถามคนก่อน", "การย้ายงานออกจากทั้งเครื่องเสี่ยงสูง ของเราจึงรอคนอนุมัติ"),
                       "The symptom moves before acting": ("อาการย้ายที่ก่อนลงมือ", "ของเราตรวจซ้ำ เห็นแจ้งเตือนเปลี่ยน จึงวางแผนใหม่"),
                       "A fault it has seen before": ("ปัญหาที่เคยเจอแล้ว", "ของเราใช้วิธีแก้ที่บันทึกไว้ครั้งก่อน ไม่ต้องถาม LLM")}},
}


def lang():
    return st.session_state.get("lang", "EN").lower()


def L(key, **kw):
    v = TXT[lang()].get(key, TXT["en"][key])
    return v.format(**kw) if isinstance(v, str) and kw else v


def fault_name(k):
    return TH_FAULTS[k] if lang() == "th" else FAULT_NAMES[k]


def ev_label(e):
    return L("ev").get(e["label"], e["label"])


def ev_text(e):
    return plain(e["tag"], e["raw"])


def css():
    st.markdown(f"""
<style>
@import url('{FONT}/400.css'); @import url('{FONT}/500.css'); @import url('{FONT}/700.css'); @import url('{FONT}/800.css');
html, body, .stApp, button, input, select, textarea, p, span, div, label {{
  font-family: "Iosevka", "Kanit", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; }}
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
.approval p {{ color: {T['deep']}; font-size: 16px !important; line-height: 1.45; }}
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
    if lang() == "th":
        return plain_th(tag, rest, m)
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


TIER_TH = {"low": "เสี่ยงต่ำ", "medium": "เสี่ยงปานกลาง", "high": "เสี่ยงสูง"}


def plain_th(tag, rest, m):
    if tag == "ALERT":
        return "แจ้งเตือนที่ " + ", ".join(re.findall(r"'([^']+)'", rest))
    if tag == "CANDIDATE" and m:
        src = "procedure store (ความจำ)" if "procedure store" in rest else ("คิวรอคนตรวจรอบสอง" if "queue" in rest else "LLM")
        return f"วางแผน {m.group(1)} ({TIER_TH.get(m.group(2), m.group(2))}) จาก {src}"
    if tag == "EXECUTE" and m:
        return f"ทำ {m.group(1)} " + ("แก้ต้นเหตุได้" if "fixed root cause" in rest else "แต่ไม่ได้แก้ต้นเหตุ")
    if tag == "RECONSIDER":
        act = rest.split("drop ")[-1].split(" and")[0]
        if "cleared" in rest:
            return f"ตรวจซ้ำ: แจ้งเตือนหายไปเอง เลยยกเลิก {act}"
        moved = ", ".join(re.findall(r"'([^']+)'", rest))
        return f"ตรวจซ้ำ: อาการย้ายไปที่ {moved} ยกเลิก {act} แล้ววางแผนใหม่"
    if tag == "HOLD":
        a = re.search(r"(\S+\(\S*\)) has low confidence \(([\d.]+)\)", rest)
        return f"พักแผน {a.group(1)} ไว้ก่อน ความมั่นใจแค่ {a.group(2)} รอหลักฐานเพิ่ม" if a else rest
    if tag == "ESCALATE":
        return f"ขอให้วิศวกร on-call อนุมัติ {rest.split(' ')[0]}"
    if tag == "HUMAN":
        verb, act = rest.split(" ", 1)
        return f"วิศวกร{'อนุมัติ' if verb == 'approved' else 'ไม่อนุมัติ'} {act}"
    if tag == "COMPILE":
        key, act = rest.split(" -> ")
        return f"บันทึก {act} เป็นวิธีแก้ของหลักฐาน {key}"
    if tag == "UNDO":
        return "ไม่ดีขึ้นในช่วงที่ย้อนได้ จึงย้อน " + rest.split("undo ")[-1]
    if tag == "POST-CHECK":
        return "ตรวจหลังทำ: ไม่ดีขึ้น " + rest.split(", ")[-1]
    if tag == "REASON":
        return rest.split(" | top:")[0]
    if tag == "REASONER":
        return "อ่านคำตอบของโมเดลไม่ได้ รอหนึ่ง tick"
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
import streamlit.components.v1 as components

# 2.5D (isometric) service map. Objects sit at fixed screen positions on an isometric grid; the layout is
# planar, so straight arrows never cross, and every label sits on a side no arrow uses.
POS = {"frontend": (125, 430), "checkout": (345, 255), "catalog": (345, 625),
       "payment": (600, 115), "inventory": (600, 265), "cart": (600, 415), "db": (840, 525)}
LABEL_AT = {"frontend": "below", "checkout": "left", "catalog": "below", "payment": "right",
            "inventory": "right", "cart": "below", "db": "below"}
NODE_OF = {s: n for n, members in NODES.items() for s in members}
C30, S30 = 0.866, 0.5
W_MAP, H_MAP = 1000, 760


def _anchor(name):
    x, y = POS[name]
    return x, y - 22


def _trim(p, q, d):
    (x1, y1), (x2, y2) = p, q
    L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5 or 1
    return x1 + (x2 - x1) * d / L, y1 + (y2 - y1) * d / L


def _edge_path(a, b):
    p, q = _anchor(a), _anchor(b)
    if False:
        tot = (q[0] - p[0]) / C30
        diff = (q[1] - p[1]) / S30
        t, u = (tot + diff) / 2, (tot - diff) / 2
        mid = (p[0] + C30 * t, p[1] + S30 * t)
        s0 = _trim(p, mid, 60)
        s1 = _trim(q, mid, 62)
        return f"M{s0[0]:.1f},{s0[1]:.1f} L{mid[0]:.1f},{mid[1]:.1f} L{s1[0]:.1f},{s1[1]:.1f}"
    s0, s1 = _trim(p, q, 64), _trim(q, p, 70)
    return f"M{s0[0]:.1f},{s0[1]:.1f} L{s1[0]:.1f},{s1[1]:.1f}"


def _iso_box(x, y, stroke, top, left, right, dash="", h=44, a=72):
    ex, ey = (C30 * a / 2, S30 * a / 2), (-C30 * a / 2, S30 * a / 2)
    cx, cy = x, y - h
    N = (cx - ex[0] - ey[0], cy - ex[1] - ey[1]); E = (cx + ex[0] - ey[0], cy + ex[1] - ey[1])
    S = (cx + ex[0] + ey[0], cy + ex[1] + ey[1]); Wp = (cx - ex[0] + ey[0], cy - ex[1] + ey[1])
    pt = lambda *ps: " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in ps)
    dn = lambda p: (p[0], p[1] + h)
    sw = f'stroke="{stroke}" stroke-width="2.4" stroke-linejoin="round"{dash}'
    out = [f'<polygon points="{pt(dn(Wp), dn(S), dn(E), (dn(E)[0] + 26, dn(E)[1] + 15), (dn(S)[0] + 26, dn(S)[1] + 15))}" '
           f'fill="{T["grid"]}" opacity=".8"/>',                                   # flat shadow on the grid
           f'<polygon points="{pt(Wp, S, dn(S), dn(Wp))}" fill="{left}" {sw}/>',
           f'<polygon points="{pt(S, E, dn(E), dn(S))}" fill="{right}" {sw}/>',
           f'<polygon points="{pt(N, E, S, Wp)}" fill="{top}" {sw}/>']
    # two drive slots on the left face, a status light on the right face
    for k, frac in enumerate((.36, .64)):
        p0 = (Wp[0] + (S[0] - Wp[0]) * .18, Wp[1] + (S[1] - Wp[1]) * .18 + h * frac)
        p1 = (Wp[0] + (S[0] - Wp[0]) * (.7 if k == 0 else .5), Wp[1] + (S[1] - Wp[1]) * (.7 if k == 0 else .5) + h * frac)
        out.append(f'<line x1="{p0[0]:.1f}" y1="{p0[1]:.1f}" x2="{p1[0]:.1f}" y2="{p1[1]:.1f}" stroke="{stroke}" stroke-width="2.4"/>')
    return out


def _iso_db(x, y, stroke, top, side):
    out = [f'<ellipse cx="{x + 22}" cy="{y + 10}" rx="46" ry="20" fill="{T["grid"]}" opacity=".8"/>']
    for k in range(3):
        yb = y - k * 24
        out.append(f'<path d="M{x - 42},{yb - 20} v20 a42,19 0 0 0 84,0 v-20" fill="{side}" stroke="{stroke}" stroke-width="2.4"/>')
        out.append(f'<ellipse cx="{x}" cy="{yb - 20}" rx="42" ry="19" fill="{top}" stroke="{stroke}" stroke-width="2.4"/>')
    return out


# The agent and the parties it talks to, drawn in the empty corners of the map.
ACTOR = {"agent": (92, 108), "llm": (455, 62), "memory": (228, 74), "oncall": (110, 652)}
BEAT_SEC = 1.35


def _actor_svg(name, active):
    x, y = ACTOR[name]
    ink = T["ink"] if active else T["line"]
    fill = T["surface"]
    lab = {"agent": "AGENT", "llm": "LLM", "memory": "MEMORY" if lang() == "en" else "ความจำ",
           "oncall": "ON-CALL"}[name]
    if name == "agent":       # a small isometric laptop
        g = (f'<polygon points="{x - 44},{y + 6} {x},{y + 31} {x + 44},{y + 6} {x},{y - 19}" fill="{T["grid"]}" stroke="{ink}" stroke-width="2.2"/>'
             f'<polygon points="{x - 44},{y + 6} {x},{y - 19} {x},{y - 69} {x - 44},{y - 44}" fill="{fill}" stroke="{ink}" stroke-width="2.2"/>'
             f'<polygon points="{x - 38},{y - 3} {x - 6},{y - 21} {x - 6},{y - 60} {x - 38},{y - 41}" fill="{T["tint"]}"/>')
    elif name == "llm":       # a cloud
        g = (f'<path d="M{x - 52},{y + 18} a22,22 0 0 1 8,-40 a28,28 0 0 1 50,-10 a24,24 0 0 1 42,22 a18,18 0 0 1 -6,28 z" '
             f'fill="{fill}" stroke="{ink}" stroke-width="2.2"/>'
             f'<text x="{x}" y="{y + 7}" text-anchor="middle" class="actor-in" fill="{ink}">LLM</text>')
        lab = ""
    elif name == "memory":    # stacked discs
        g = "".join(f'<path d="M{x - 26},{y - k * 14 - 8} v8 a26,9 0 0 0 52,0 v-8" fill="{fill}" stroke="{ink}" stroke-width="2"/>'
                    f'<ellipse cx="{x}" cy="{y - k * 14 - 8}" rx="26" ry="9" fill="{T["grid"]}" stroke="{ink}" stroke-width="2"/>'
                    for k in range(3))
    else:                     # on-call person
        g = (f'<circle cx="{x}" cy="{y - 40}" r="14" fill="{fill}" stroke="{ink}" stroke-width="2.4"/>'
             f'<path d="M{x - 26},{y + 6} q0,-30 26,-30 q26,0 26,30 z" fill="{fill}" stroke="{ink}" stroke-width="2.4"/>')
    text = (f'<text x="{x}" y="{y + 52}" text-anchor="middle" class="actor" fill="{ink}">{lab}</text>' if lab else "")
    return g + text


def _beats(events, upto):
    """Turn the events of this step into short trips between the agent, its helpers and the services.
    Each trip has a card: a coloured header (what kind of message) and one or two body lines (what it says)."""
    now = [e for e in events if e["frame"] == upto]
    th = lang() == "th"
    w = (lambda en, t: t if th else en)
    tier_txt = {"low": w("low risk", "เสี่ยงต่ำ"), "medium": w("medium risk", "เสี่ยงปานกลาง"), "high": w("high risk", "เสี่ยงสูง")}
    out = []
    for e in now:
        tag, raw = e["tag"], e["raw"]
        act = raw.split(" ")[0]
        tier = re.search(r"\[(\w+)\]", raw)
        tier = tier_txt.get(tier.group(1), "") if tier else ""
        tgt = act.split("(")[1].rstrip(")") if "(" in act else ""
        tpos = (POS[NODES[tgt][0]] if tgt in NODES else POS.get(tgt))
        A = ACTOR["agent"]
        if tag == "ALERT":
            svcs = re.findall(r"'([^']+)'", raw)
            if svcs:
                more = f" +{len(svcs) - 1}" if len(svcs) > 1 else ""
                out.append(dict(a=POS[svcs[0]], b=A, head=w("ALERT", "แจ้งเตือน"), body=[svcs[0] + more],
                                col=T["alert"]))
        elif tag == "CANDIDATE":
            if "procedure store" in raw:
                out.append(dict(a=ACTOR["memory"], b=A, head=w("REUSE FIX", "ใช้วิธีเดิม"), body=[act, tier], col=T["memory"]))
            elif "queue" in raw:
                out.append(dict(a=A, b=A, head=w("ASK AGAIN", "ถามคนอีกรอบ"), body=[act], col=T["human"]))
            else:
                out.append(dict(a=A, b=ACTOR["llm"], head=w("ASK LLM", "ถาม LLM"),
                                body=[w("alerts, logs, traces", "แจ้งเตือน, log, trace")], col=T["ink"]))
                out.append(dict(a=ACTOR["llm"], b=A, head=w("PLAN", "แผน"), body=[act, tier], col=T["ink"]))
        elif tag == "HOLD":
            c = re.search(r"\(([\d.]+)\)", raw)
            out.append(dict(a=A, b=A, head=w("HOLD", "พักแผน"),
                            body=[w("not sure yet", "ยังไม่มั่นใจ") + (f" ({c.group(1)})" if c else "")], col=T["pending"]))
        elif tag == "RECONSIDER":
            svcs = re.findall(r"'([^']+)'", raw)
            cleared = "cleared" in raw
            out.append(dict(a=POS[svcs[0]] if svcs else A, b=A, head=w("RE-CHECK", "ตรวจซ้ำ"),
                            body=[w("alerts cleared", "แจ้งเตือนหายเอง") if cleared else w("symptom moved", "อาการย้ายที่")],
                            col=T["pending"], result=(w("PLAN DROPPED", "ยกเลิกแผน"), T["pending"])))
        elif tag == "ESCALATE":
            out.append(dict(a=A, b=ACTOR["oncall"], head=w("APPROVE?", "อนุมัติไหม?"), body=[act, tier_txt["high"]],
                            col=T["human"]))
        elif tag == "HUMAN":
            ok = raw.startswith("approved")
            out.append(dict(a=ACTOR["oncall"], b=A, head=w("APPROVED", "อนุมัติ") if ok else w("REJECTED", "ไม่อนุมัติ"),
                            body=[raw.split(" ", 1)[1]], col=T["human"] if ok else T["alert"]))
        elif tag == "EXECUTE" and tpos:
            fixed = "fixed root cause" in raw
            out.append(dict(a=A, b=tpos, head=w("RUN", "สั่งทำ"), body=[act, tier], col=T["focus"],
                            result=((w("FIXED", "แก้ได้"), T["resolved"]) if fixed else (w("NO EFFECT", "ไม่ได้ผล"), T["alert"]))))
        elif tag == "UNDO" and tpos:
            out.append(dict(a=A, b=tpos, head=w("UNDO", "ย้อนกลับ"), body=[act], col=T["pending"]))
        elif tag == "COMPILE":
            key, fix = raw.split(" -> ")
            out.append(dict(a=A, b=ACTOR["memory"], head=w("SAVE FIX", "บันทึกวิธีแก้"), body=[fix, key], col=T["memory"]))
    for bt in out:
        bt["body"] = [x for x in bt["body"] if x]
    return out


def _trim_pt(p, q, d):
    (x1, y1), (x2, y2) = p, q
    L = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** .5 or 1
    return x2 - (x2 - x1) * d / L, y2 - (y2 - y1) * d / L


def _spot(pt, obj_svg):
    for name, p in ACTOR.items():
        if p == pt:
            return _actor_svg(name, True)
    for s, p in POS.items():
        if p == pt:
            return obj_svg(s)
    return ""


def card_height(bt):
    return 52 + 30 * len(bt["body"]) + (12 if bt["body"] else 0) + 22


def _card(bt, flip=False):
    """Two-part card drawn above (0, 0), or below it when flip: coloured header, white body, a tail at the link."""
    col, head, body = bt["col"], bt["head"], bt["body"]
    wh = 16.5 * len(head) + 40
    wb = max([13.2 * len(x) + 32 for x in body] or [0])
    w = max(wh, wb, 150)
    hb = 30 * len(body) + (12 if body else 0)
    top = 22 if flip else -(52 + hb + 22)
    tail = (f'<path d="M-13,{top + 2} L0,{top - 18} L13,{top + 2} z" fill="{col}"/>' if flip else
            f'<path d="M-13,{top + 52 + hb - 2:.0f} L0,{top + 52 + hb + 18:.0f} L13,{top + 52 + hb - 2:.0f} z" fill="{col}"/>')
    parts = [f'<rect x="{-w / 2:.0f}" y="{top:.0f}" width="{w:.0f}" height="{52 + hb:.0f}" fill="#fff" stroke="{col}" stroke-width="3"/>',
             f'<rect x="{-w / 2:.0f}" y="{top:.0f}" width="{w:.0f}" height="48" fill="{col}"/>',
             f'<text x="0" y="{top + 33:.0f}" text-anchor="middle" class="card-h">{html.escape(head)}</text>',
             tail]
    for i, line in enumerate(body):
        parts.append(f'<text x="{-w / 2 + 16:.0f}" y="{top + 82 + 30 * i:.0f}" class="card-b">{html.escape(line)}</text>')
    return "".join(parts)


def _beats_svg(beats, speed, obj_svg):
    """One trip at a time: the two ends light up, the path draws itself, a labelled card rides it and parks
    just short of the target, then a result chip appears. Earlier trips clear away; the last one stays."""
    beat = BEAT_SEC / speed
    o = []
    for k, bt in enumerate(beats):
        last = k == len(beats) - 1
        t0 = 0.4 + k * beat
        t_end = t0 + beat - .15
        hide = "" if last else f'<set attributeName="opacity" to="0" begin="{t_end:.2f}s"/>'
        col = bt["col"]
        # spotlight: full-strength copies of both ends on top of the faded map
        ends = {bt["a"], bt["b"]}
        o.append(f'<g opacity="0"><set attributeName="opacity" to="1" begin="{t0:.2f}s"/>{hide}'
                 + "".join(_spot(p, obj_svg) for p in ends) + '</g>')
        (x1, y1), (x2, y2) = bt["a"], bt["b"]
        y1, y2 = y1 - 30, y2 - 30
        if (x1, y1) != (x2, y2):
            sx, sy = _trim_pt((x2, y2), (x1, y1), 0) if False else (x1, y1)
            ex, ey = _trim_pt((x1, y1), (x2, y2), 78)          # stop short of the target
            L = ((ex - sx) ** 2 + (ey - sy) ** 2) ** .5
            d = f"M{sx:.0f},{sy:.0f} L{ex:.0f},{ey:.0f}"
            o.append(f'<path d="{d}" stroke="{col}" stroke-width="5" fill="none" stroke-linecap="round" '
                     f'stroke-dasharray="{L:.0f}" stroke-dashoffset="{L:.0f}" opacity="0">'
                     f'<set attributeName="opacity" to="1" begin="{t0:.2f}s"/>'
                     f'<animate attributeName="stroke-dashoffset" from="{L:.0f}" to="0" begin="{t0:.2f}s" '
                     f'dur="{.5 / speed:.2f}s" fill="freeze"/>{hide}</path>')
            o.append(f'<circle cx="{ex:.0f}" cy="{ey:.0f}" r="9" fill="{col}" opacity="0">'
                     f'<set attributeName="opacity" to="1" begin="{t0 + .5 / speed:.2f}s"/>{hide}</circle>')
            mx, my = (sx + ex) / 2, (sy + ey) / 2     # the card rides to the middle of the link and waits there
            dm = f"M{sx:.0f},{sy:.0f} L{mx:.0f},{my:.0f}"
            o.append(f'<g opacity="0"><set attributeName="opacity" to="1" begin="{t0:.2f}s"/>'
                     f'<animateMotion path="{dm}" begin="{t0:.2f}s" dur="{.7 / speed:.2f}s" fill="freeze" '
                     f'calcMode="spline" keyTimes="0;1" keySplines=".4 0 .2 1"/>{hide}' + _card(bt, my - card_height(bt) < 8) + '</g>')
            o.append(f'<circle r="11" fill="{col}" opacity="0" stroke="#fff" stroke-width="3">'
                     f'<set attributeName="opacity" to="1" begin="{t0:.2f}s"/>'
                     f'<animateMotion path="{d}" begin="{t0:.2f}s" dur="{.9 / speed:.2f}s" fill="freeze"/>{hide}</circle>')
        else:
            flip = y1 - 40 - card_height(bt) < 8
            o.append(f'<g opacity="0" transform="translate({x1 + 40},{y1 + (20 if flip else -40)})"><set attributeName="opacity" '
                     f'to="1" begin="{t0:.2f}s"/>{hide}' + _card(bt, flip) + '</g>')
        if bt.get("result"):
            txt, rc = bt["result"]
            rw = 14.5 * len(txt) + 40
            o.append(f'<g opacity="0" transform="translate({x2},{y2 - 96})"><set attributeName="opacity" to="1" '
                     f'begin="{t0 + .9 / speed:.2f}s"/>{hide}'
                     f'<rect x="{-rw / 2:.0f}" y="-23" width="{rw:.0f}" height="46" fill="{T["surface"]}" stroke="{rc}" stroke-width="3.5"/>'
                     f'<circle cx="{-rw / 2 + 19:.0f}" cy="0" r="8" fill="{rc}"/>'
                     f'<text x="{-rw / 2 + 36:.0f}" y="8" class="chip-r" fill="{rc}">{html.escape(txt)}</text></g>')
    return "".join(o)


def beat_seconds(events, upto, speed=1.0):
    n = len(_beats(events, upto))
    return 1.6 / speed if n == 0 else 0.4 + n * BEAT_SEC / speed + 1.2


def map_html(obs, events, upto, height=500, motion=True, speed=1.0):
    """Animated 2.5D service map, drawn in its own frame so SVG animation runs."""
    alerting = {s for s, _ in obs["alerts"]} if obs else set()
    m = obs["metrics"] if obs else {}
    now = [e for e in events if e["frame"] == upto]
    acts = [e["raw"].split(" ")[0] for e in events if e["tag"] == "EXECUTE" and e["frame"] <= upto]
    drained = {a.split("(")[1].rstrip(")") for a in acts if a.startswith("drain_node")}
    acting = next((e for e in reversed(now) if e["tag"] in ("EXECUTE", "ESCALATE", "HUMAN")), None)
    if acting is None:   # still waiting for the on-call answer: keep the card up
        past = [e for e in events if e["frame"] <= upto and e["tag"] in ("ESCALATE", "HUMAN")]
        if past and past[-1]["tag"] == "ESCALATE":
            acting = past[-1]
    W, H = W_MAP, H_MAP
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" height="100%" preserveAspectRatio="xMidYMid meet" role="img" '
         f'aria-label="Service map. Orange services are alerting." xmlns="http://www.w3.org/2000/svg">',
         f'<defs><pattern id="iso" width="69.28" height="40" patternUnits="userSpaceOnUse">'
         f'<path d="M0,20 L34.64,0 L69.28,20 L34.64,40 Z" fill="none" stroke="{T["grid"]}" stroke-width="1"/></pattern>'
         f'<marker id="a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto">'
         f'<path d="M0,0L10,5L0,10z" fill="{T["focus"]}"/></marker>'
         f'<marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="9" markerHeight="9" orient="auto">'
         f'<path d="M0,0L10,5L0,10z" fill="{T["alert"]}"/></marker></defs>',
         f'<rect width="{W}" height="{H}" fill="{T["surface"]}"/><rect width="{W}" height="{H}" fill="url(#iso)"/>']
    beats = _beats(events, upto) if motion else []
    o.append('<g id="base">')
    if beats:   # everything not in the current trip fades back, like the reference explorer
        o.append('<animate attributeName="opacity" from="1" to=".16" begin=".05s" dur=".35s" fill="freeze"/>')
    # arrows first, so objects sit on top of them
    k = 0
    for src, deps in DEPS.items():
        for d in deps:
            bad = d in alerting
            col = T["alert"] if bad else T["focus"]
            o.append(f'<path id="e{k}" d="{_edge_path(src, d)}" class="edge {"bad" if bad else ""}" stroke="{col}" '
                     f'marker-end="url(#{"ar" if bad else "a"})"/>')
            dur = 1.0 if bad else 2.0
            if motion:
                o.append(f'<circle r="{6 if bad else 5}" fill="{col}" class="pkt"><animateMotion dur="{dur}s" '
                         f'repeatCount="indefinite"><mpath href="#e{k}"/></animateMotion></circle>')
            k += 1
    for name in ACTOR:
        o.append(_actor_svg(name, name == "agent"))

    def obj_svg(s):
        o = []
        x, y = POS[s]
        bad = s in alerting
        down = NODE_OF.get(s, "") in drained
        stroke = T["alert"] if bad else T["ink"]
        top, left, right = ((T["alert_tint"], "#FFF6F1", "#F7D3C0") if bad else (T["grid"], T["surface"], T["tint"]))
        if bad:
            o.append(f'<polygon points="{x},{y - 80} {x + 96},{y - 22} {x},{y + 36} {x - 96},{y - 22}" fill="none" '
                     f'stroke="{T["alert"]}" stroke-width="2.5" class="ring"/>')
        if s == "db":
            o += _iso_db(x, y, stroke, top, left)
        else:
            o += _iso_box(x, y, stroke, top, left, right, ' stroke-dasharray="6 5"' if down else "")
            lx, ly = x + 31.2, y - 4.5                 # centre of the right face
            o.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6" fill="{T["alert"] if bad else T["resolved"]}"/>')
        where = LABEL_AT[s]
        name_col = T["alert"] if bad else T["deep"]
        metric = f'{m[s]["error_rate"] * 100:.0f}% err  {m[s]["p99_ms"]} ms' if m else ""
        node = NODE_OF.get(s, "")
        tag = f"{node}{' drained' if down else ''}" if node and s != "db" else ""
        if where == "below":
            tx, ty, anc = x - (22 if s == "frontend" else 0), y + 66 if s != "db" else y + 44, "middle"
        elif where == "above":
            tx, ty, anc = x, y - 146, "middle"
        elif where == "left":
            tx, ty, anc = x - 78, y - 40, "end"
        else:
            tx, ty, anc = x + 78, y - 40, "start"
        o.append(f'<text x="{tx}" y="{ty}" text-anchor="{anc}" class="name" fill="{name_col}">{s}</text>')
        if metric:
            o.append(f'<text x="{tx}" y="{ty + 25}" text-anchor="{anc}" class="metric" fill="{name_col if bad else T["muted"]}">{metric}</text>')
        if tag:
            o.append(f'<text x="{tx}" y="{ty + 47}" text-anchor="{anc}" class="node" fill="{T["muted"]}">{tag}</text>')
        return "".join(o)

    # objects, back to front
    for s in sorted(POS, key=lambda n: POS[n][1]):
        o.append(obj_svg(s))
    o.append('</g>')
    if beats:
        o.append(_beats_svg(beats, speed, obj_svg))
        acting = None
    if acting is not None:
        target = acting["raw"].split(" ")[0]
        tgt = target.split("(")[1].rstrip(")") if "(" in target else ""
        col = T["human"] if acting["tag"] in ("ESCALATE", "HUMAN") else T["focus"]
        word = L("asking_oncall") if acting["tag"] in ("ESCALATE", "HUMAN") else L("agent_acts")
        boxes = [POS[x] for x in NODES[tgt]] if tgt in NODES else ([POS[tgt]] if tgt in POS else [])
        for (bx, by) in boxes:
            if tgt == "db":
                o.append(f'<rect x="{bx - 62}" y="{by - 112}" width="124" height="138" fill="none" stroke="{col}" '
                         f'stroke-width="3" stroke-dasharray="8 6" class="flow"/>')
                continue
            o.append(f'<polygon points="{bx},{by - 86} {bx + 104},{by - 22} {bx},{by + 42} {bx - 104},{by - 22}" '
                     f'fill="none" stroke="{col}" stroke-width="3" stroke-dasharray="8 6" class="flow"/>')
        if boxes:
            bx, by = min(boxes, key=lambda p: p[0])
            body = target
            w = max(13.5 * len(word), 12.2 * len(body)) + 28
            cx, cy = 16, 16                      # the top-left corner is always empty in this layout
            o.append(f'<g class="pop"><rect x="{cx}" y="{cy}" width="{w}" height="78" fill="{T["surface"]}" stroke="{col}" stroke-width="2.5"/>'
                     f'<rect x="{cx}" y="{cy}" width="{w}" height="34" fill="{col}"/>'
                     f'<text x="{cx + w / 2}" y="{cy + 24}" text-anchor="middle" class="chip">{word}</text>'
                     f'<text x="{cx + 14}" y="{cy + 62}" class="body" fill="{T["deep"]}">{html.escape(body)}</text></g>')
    o.append("</svg>")
    css_ = f"""<style>@import url('https://fonts.googleapis.com/css2?family=Kanit:wght@400;500;600&display=swap'); @import url('{FONT}/500.css'); @import url('{FONT}/600.css'); @import url('{FONT}/800.css');
html,body{{margin:0;height:100%;overflow:hidden;background:{T['surface']};font-family:Iosevka,"Kanit",ui-monospace,monospace}}
.edge{{fill:none;stroke-width:3;stroke-dasharray:9 7;animation:flow 1.1s linear infinite}}
.edge.bad{{stroke-width:3.4;animation-duration:.55s}}
@keyframes flow{{to{{stroke-dashoffset:-32}}}}
.flow{{animation:flow .9s linear infinite}}
.ring{{transform-box:fill-box;transform-origin:center;animation:ring 1.6s ease-out infinite}}
@keyframes ring{{0%{{opacity:.8;transform:scale(.8)}}100%{{opacity:0;transform:scale(1.18)}}}}
.pop{{animation:pop .35s ease-out}} @keyframes pop{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1}}}}
.name{{font-size:27px;font-weight:800;letter-spacing:.02em}} .metric{{font-size:20px;font-weight:600}}
.node{{font-size:17px;font-weight:600}} .chip{{font-size:19px;font-weight:800;fill:#fff;letter-spacing:.06em}}
.body{{font-size:20px;font-weight:600}}
.dimmed{{opacity:.28}} .card{{font-size:19px;font-weight:800;fill:#fff;letter-spacing:.03em}}
.card-h{{font-size:27px;font-weight:800;fill:#fff;letter-spacing:.08em}} .card-b{{font-size:22px;font-weight:600;fill:{T['deep']}}}
.chip-r{{font-size:22px;font-weight:800;letter-spacing:.04em}} .actor{{font-size:18px;font-weight:800;letter-spacing:.08em}}
.actor-in{{font-size:20px;font-weight:800}}
{"" if motion else "*{animation:none!important}"}
</style>"""
    return css_ + "".join(o)


def show_map(obs, events, upto, height=500):
    components.html(map_html(obs, events, upto, motion=st.session_state.get("motion", True),
                             speed=st.session_state.get("speed", 1.0)), height=height, scrolling=False)


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
            f'<p class="figure">{fmt.format(before)} <span class="arrow">{"to" if lang() == "en" else "เป็น"}</span> {fmt.format(after)}</p>'
            f'<p class="muted" style="font-size:13px;margin:0">{note}</p></div>')


RES_TH = {"Actions that did not help": "action ที่ไม่ช่วย", "Outage the agent caused": "ระบบล่มเพราะ agent", "Time to recover": "เวลากู้คืน", "LLM calls": "การเรียก LLM", "Actions that did not help, per incident": "action ที่ไม่ช่วย ต่อเหตุการณ์", "Where they happened": "เกิดกับปัญหาแบบไหน", "Memory: LLM calls on a repeated fault": "ความจำ: การเรียก LLM เมื่อเจอปัญหาซ้ำ", "Scripted reasoner: ablation and sensitivity (30 episodes each)": "ตัวคิดแบบสคริปต์: ablation และ sensitivity (ชุดละ 30 episode)", "Clear evidence": "หลักฐานชัด", "Noisy evidence": "หลักฐานมีสัญญาณหลอก"}


def R(t):
    return RES_TH.get(t, t) if lang() == "th" else t


def results_tab():
    folders = [d for d in ("results/llm_clear", "results/llm_noisy") if os.path.exists(os.path.join(d, "incidents.csv"))]
    if not folders:
        st.markdown('<div class="panel"><p class="ptitle">No LLM results yet</p><p>Run run_experiments.py with '
                    '--reasoner llm and the results appear here.</p></div>', unsafe_allow_html=True)
        return
    df = pd.concat([load_incidents(f) for f in folders])
    n_ep = df.episode.nunique()
    lead = (f"B1 against our agent, muse-glimmer-30b on the same {n_ep * 16} incidents per setting "
            f"({n_ep} episodes of 16). Bars show the mean, lines the 95% bootstrap interval.") if lang() == "en" else (
            f"เทียบ B1 กับ agent ของเรา ใช้ muse-glimmer-30b กับเหตุการณ์ชุดเดียวกัน {n_ep * 16} เหตุการณ์ต่อชุด "
            f"({n_ep} episode ละ 16) แท่งคือค่าเฉลี่ย เส้นคือช่วงความเชื่อมั่น 95% แบบ bootstrap")
    st.markdown(f'<p class="lead">{lead}</p>', unsafe_allow_html=True)
    for h in headline(df):
        st.markdown(f'<p class="ptitle" style="margin-top:6px">{R(h["setting"])}</p>', unsafe_allow_html=True)
        c = st.columns(4)
        b, o = h["unneeded"]
        c[0].markdown(figure_card(R("Actions that did not help"), b, o, "{:.2f}",
                                  (f"per incident, {100 * (o - b) / b:+.0f}%" if lang() == "en" else f"ต่อเหตุการณ์ {100 * (o - b) / b:+.0f}%")), unsafe_allow_html=True)
        b, o = h["damage"]
        c[1].markdown(figure_card(R("Outage the agent caused"), b, o, "{}", "ticks in total (1 tick = 30 s)" if lang() == "en" else "tick รวม (1 tick = 30 วินาที)"),
                      unsafe_allow_html=True)
        b, o = h["mttr"]
        c[2].markdown(figure_card(R("Time to recover"), b, o, "{:.1f}", f"minutes, {o - b:+.1f} min: the price of checking" if lang() == "en" else f"นาที ช้าลง {o - b:+.1f} นาที เป็นราคาของการตรวจซ้ำ"),
                      unsafe_allow_html=True)
        b, o = h["calls"]
        c[3].markdown(figure_card(R("LLM calls"), b, o, "{:.2f}", f"per incident, {100 * (o - b) / b:+.0f}% thanks to memory" if lang() == "en" else f"ต่อเหตุการณ์ {100 * (o - b) / b:+.0f}% เพราะมีความจำ"),
                      unsafe_allow_html=True)

    st.markdown('<p class="ptitle" style="margin-top:14px">' + R("Actions that did not help, per incident") + '</p>', unsafe_allow_html=True)
    st.altair_chart(alt_theme(bar_ci(mean_ci(df, "unnecessary"), "per incident"), 150), use_container_width=True)

    l, r = st.columns(2)
    with l:
        st.markdown('<p class="ptitle">' + R("Where they happened") + '</p>', unsafe_allow_html=True)
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
        st.markdown('<p class="ptitle">' + R("Time to recover") + '</p>', unsafe_allow_html=True)
        st.altair_chart(alt_theme(bar_ci(mean_ci(df, "ttr_min"), "minutes"), 150), use_container_width=True)
        st.markdown('<p class="ptitle">' + R("Memory: LLM calls on a repeated fault") + '</p>', unsafe_allow_html=True)
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

    runs = [("qwen2.5 7B", "results/llm_noisy_7b"), ("qwen3.8 27B", "results/llm_noisy_qwen"),
            ("muse-glimmer 30B", "results/llm_noisy")]
    runs = [(n, f) for n, f in runs if os.path.exists(os.path.join(f, "incidents.csv"))]
    if len(runs) > 1:
        st.markdown('<p class="ptitle" style="margin-top:14px">' + ("Does it matter how strong the LLM is? (noisy evidence)"
                    if lang() == "en" else "ถ้า LLM เก่งไม่เท่ากัน ผลต่างกันไหม (หลักฐานมีสัญญาณหลอก)") + '</p>',
                    unsafe_allow_html=True)
        rows = []
        for name, f in runs:
            d = pd.read_csv(os.path.join(f, "incidents.csv"))
            for arm in ("B1-ReAct", "Ours"):
                g = d[d.agent == arm]
                lo, hi = bootstrap(g.unnecessary.astype(float))
                rows.append({"model": name, "agent": ARM_LABEL[arm], "did not help": g.unnecessary.mean(), "lo": lo, "hi": hi,
                             "outage ticks": int(g.collateral_ticks.sum()), "recover (min)": round(g.ttr_min.mean(), 2),
                             "resolved": round(g.resolved.mean(), 3)})
        df_m = pd.DataFrame(rows)
        l2, r2 = st.columns([1.3, 1])
        with l2:
            base = alt.Chart(df_m).encode(
                x=alt.X("model:N", sort=[n for n, _ in runs], title=None, axis=alt.Axis(labelAngle=0)),
                xOffset="agent:N",
                color=alt.Color("agent:N", scale=alt.Scale(domain=["B1 ReAct", "Ours"],
                                                           range=[ARM_COLOUR["B1 ReAct"], ARM_COLOUR["Ours"]])))
            ch = alt.layer(base.mark_bar().encode(y=alt.Y("did not help:Q", title="did not help / incident")),
                           base.mark_rule(color=T["deep"], strokeWidth=2).encode(y="lo:Q", y2="hi:Q"))
            st.altair_chart(alt_theme(ch, 260), use_container_width=True)
        with r2:
            st.dataframe(df_m[["model", "agent", "did not help", "outage ticks", "recover (min)", "resolved"]].round(2),
                         hide_index=True, use_container_width=True)

    with st.expander(R("Scripted reasoner: ablation and sensitivity (30 episodes each)")):
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
    st.markdown('<p class="muted" style="font-size:13px">' + ("Every chart has a menu (three dots, top right) to save it "
                "as PNG or SVG for the slides and the paper." if lang() == "en" else "ทุกกราฟกดเมนูจุดสามจุดมุมขวาบนเพื่อบันทึกเป็น PNG หรือ SVG ไปใช้ในสไลด์และ paper ได้")
                + '</p>', unsafe_allow_html=True)


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


def side(run, upto, title, key="panel-side"):
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
    with st.container(border=True, key=key):
        st.markdown(f'<div class="sidehead"><p class="ptitle">{title}</p>'
                    f'<span class="badge" style="color:{badge[1]};border-color:{badge[1]}">{badge[0]}</span></div>',
                    unsafe_allow_html=True)
        show_map(obs, run["events"], min(upto, n - 1), height=440)
        items = "".join(f'<li><span class="num" style="background:{e["colour"]}">{k}</span><span>'
                        f'<b style="color:{e["colour"]}">{e["label"]}</b> {html.escape(e["text"])}</span></li>'
                        for k, e in enumerate(events, 1)) or '<li><span></span><span class="muted">Nothing yet.</span></li>'
        score = ""
        if finished:
            score = (f'<div class="score"><div><p class="label">Actions</p><p class="big">{res.executed}</p>'
                     f'<p class="muted">{res.unnecessary} did not help</p></div>'
                     f'<div><p class="label">Outage caused</p><p class="big">{res.collateral_ticks}</p><p class="muted">ticks</p></div>'
                     f'<div><p class="label">Recovered in</p><p class="big">'
                     f'{"%.1f" % res.ttr_min if res.ttr_min is not None else "no"}</p><p class="muted">minutes</p></div></div>')
        st.markdown(f'<ul class="steps big">{items}</ul>{score}', unsafe_allow_html=True)


def timeline(a, b, upto, last):
    """One strip per agent: a dot for every step, a line for now."""
    W, H, x0, x1 = 1000, 96, 120, 980
    sx = lambda f: x0 + (x1 - x0) * f / max(1, last)
    o = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Timeline of both agents">']
    for k in range(last + 1):
        o.append(f'<line x1="{sx(k)}" y1="14" x2="{sx(k)}" y2="86" stroke="{T["grid"]}" stroke-width="1"/>')
        if k % 2 == 0:
            o.append(f'<text x="{sx(k)}" y="96" text-anchor="middle" font-size="13" fill="{T["muted"]}">{k * TICK_SECONDS / 60:.0f}'
                     f'{" min" if k == 0 else ""}</text>' if k * TICK_SECONDS % 60 == 0 else "")
    for row, (run, label) in enumerate(((a, "B1"), (b, "Ours"))):
        y = 32 + row * 34
        o.append(f'<text x="0" y="{y + 6}" font-size="17" font-weight="800" fill="{T["ink"]}">{label}</text>'
                 f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="{T["line"]}" stroke-width="2"/>')
        seen = {}
        for e in run["events"]:
            if e["tag"] == "REASON":
                continue
            j = seen.get(e["frame"], 0)
            seen[e["frame"]] = j + 1
            op = 1 if e["frame"] <= upto else .25
            o.append(f'<circle cx="{sx(e["frame"]) + j * 13}" cy="{y}" r="7" fill="{e["colour"]}" opacity="{op}"/>')
    o.append(f'<rect x="{sx(upto) - 2}" y="8" width="4" height="80" fill="{T["focus"]}"/></svg>')
    st.markdown(f'<div class="panel" style="padding:8px 14px">{"".join(o)}</div>', unsafe_allow_html=True)


def details_panel(a, b, upto, story):
    """Right column: what each agent is doing at this moment, in words."""
    def block(run, who):
        evs = [e for e in run["events"] if e["frame"] <= upto and e["tag"] != "REASON"]
        res, n = run["res"], len(run["frames"])
        if upto >= n - 1:
            outcome = L("recovered_in", m=res.ttr_min) if res.resolved else L("did_not_recover")
            body = (f'<p class="det-last" style="color:{T["resolved"] if res.resolved else T["alert"]}">{outcome}</p>'
                    f'<table class="kv"><tr><td>{L("actions")}</td><td>{L("actions_val", n=res.executed, u=res.unnecessary)}</td></tr>'
                    f'<tr><td>{L("outage")}</td><td>{L("outage_val", m=res.collateral_ticks * TICK_SECONDS // 60, t=res.collateral_ticks)}'
                    f'</td></tr></table>')
        elif evs:
            e = evs[-1]
            body = (f'<p class="det-last"><span class="det-tag" style="background:{e["colour"]}">{ev_label(e)}</span> '
                    f'{html.escape(ev_text(e))}</p>')
        else:
            body = f'<p class="det-last muted">{L("nothing_yet")}</p>'
        reason = ""
        if who != "B1":
            raw = next((e["raw"] for e in reversed(run["events"]) if e["frame"] <= upto and e["tag"] == "REASON"), "")
            if raw:
                reason = f'<p class="det-why"><b>{L("diagnosis")}</b> {html.escape(raw.split(" | top:")[0][:260])}</p>'
        return f'<div class="det-block"><p class="det-who">{who}</p>{body}{reason}</div>'

    why = L("stories").get(story["title"], (story["title"], story["why"]))[1]
    st.markdown(f'<div class="panel details"><p class="ptitle">{L("details")}</p>'
                f'<p class="det-time">{L("clock", m=upto * TICK_SECONDS / 60)}</p>'
                f'{block(a, "B1")}{block(b, L("ours"))}'
                f'<p class="det-story">{html.escape(why)} {L("hidden", f=fault_name(story["fault"]), t=story["target"])}</p></div>',
                unsafe_allow_html=True)


def flow_log(a, b, upto):
    rows = []
    for who, run in (("B1", a), (L("ours"), b)):
        for e in run["events"]:
            if e["frame"] <= upto and e["tag"] != "REASON":
                rows.append((e["frame"], who, e))
    rows.sort(key=lambda r: (r[0], r[1] != "B1"))
    legend = "".join(f'<span class="leg"><span class="dot" style="background:{c}"></span>{L("ev").get(l, l)}</span>'
                     for l, c in LEGEND.items())
    cols = L("cols")
    body = "".join(
        f'<tr class="{"now" if f == upto else ""}"><td>{k}</td><td>{L("min", m=f * TICK_SECONDS / 60)}</td><td><b>{who}</b></td>'
        f'<td><span class="dot" style="background:{e["colour"]}"></span>{ev_label(e)}</td><td>{html.escape(ev_text(e))}</td></tr>'
        for k, (f, who, e) in reversed(list(enumerate(rows, 1))))
    body = body or f'<tr><td colspan="5" class="muted">{L("press_next")}</td></tr>'
    st.markdown(f'<div class="panel flowlog"><div class="flowhead"><p class="ptitle">{L("log")}</p>{legend}'
                f'<span class="muted" style="font-size:13px">{L("newest")}</span></div>'
                f'<div class="flowbody"><table class="log"><tr><th style="width:60px">{cols[0]}</th><th style="width:100px">{cols[1]}</th>'
                f'<th style="width:80px">{cols[2]}</th><th style="width:130px">{cols[3]}</th><th>{cols[4]}</th></tr>{body}'
                f'</table></div></div>', unsafe_allow_html=True)


def replay_tab():
    folders = [d for d in ("results/llm_noisy", "results/llm_clear") if os.path.exists(os.path.join(d, "llm_cache.jsonl"))]
    if not folders:
        st.markdown(f'<div class="panel"><p class="ptitle">{L("no_runs_t")}</p><p>{L("no_runs")}</p></div>',
                    unsafe_allow_html=True)
        return
    stories = []
    for f in folders:
        for s_ in find_stories(f):
            s_["folder"] = f
            if s_["title"] not in [x["title"] for x in stories]:
                stories.append(s_)
    c = st.columns([2.6, 0.8, 1, 0.9, 0.9, 0.65], vertical_alignment="center")
    pick = c[0].selectbox("Incident", list(range(len(stories))), label_visibility="collapsed",
                          format_func=lambda k: L("stories").get(stories[k]["title"], (stories[k]["title"],))[0])
    s_ = stories[pick]
    key = (s_["folder"], s_["episode"], s_["index"])
    if ss.get("pair_key") != key:
        noise = 0.5 if "noisy" in s_["folder"] else 0.0
        ss.pair = (replay(s_["folder"], s_["episode"], s_["index"], "B1-ReAct", noise),
                   replay(s_["folder"], s_["episode"], s_["index"], "Ours", noise))
        ss.pair_key, ss.t, ss.playing = key, 0, False
    a, b = ss.pair
    last = max(len(a["frames"]), len(b["frames"])) - 1
    if c[1].button(L("back"), use_container_width=True):
        ss.t, ss.playing = max(0, ss.t - 1), False
    if c[2].button(L("next"), type="primary", use_container_width=True):
        ss.t, ss.playing = min(last, ss.t + 1), False
    if c[3].button(L("pause") if ss.get("playing") else L("play"), use_container_width=True, key="playbtn"):
        ss.playing = not ss.get("playing")
        if ss.playing and ss.t >= last:
            ss.t = 0
        ss.last_tick = time.time()
        st.rerun()
    if c[4].button(L("start_over"), use_container_width=True):
        ss.t, ss.playing = 0, False
    c[5].selectbox("Speed", [0.5, 1.0, 1.5, 2.0], index=1, key="speed", label_visibility="collapsed",
                   format_func=lambda v: f"{v:g}×")

    sp = ss.get("speed", 1.0)
    every = (max(beat_seconds(a["events"], ss.t, sp), beat_seconds(b["events"], ss.t, sp))
             if ss.get("motion", True) else 1.6 / sp) if ss.get("playing") else None

    @st.fragment(run_every=every)
    def stage():
        # On a timer tick, move one step; the map, details and log below all redraw for the new step.
        if ss.get("playing") and time.time() - ss.get("last_tick", 0) >= (every or 0) - .05:
            if ss.t < last:
                ss.t += 1
                ss.last_tick = time.time()
            else:
                ss.playing = False
                st.rerun()
        st.markdown(f'<p class="clock" style="margin:0">{L("clock", m=ss.t * TICK_SECONDS / 60)}</p>', unsafe_allow_html=True)
        left, mid, right = st.columns([1.5, 1.5, 1.05])
        for col, run, title, k in ((left, a, L("b1_title"), "panel-b1"), (mid, b, L("ours_title"), "panel-ours")):
            with col:
                n = len(run["frames"])
                obs = run["frames"][min(ss.t, n - 1)] if n else None
                res = run["res"]
                if ss.t >= n - 1:
                    badge = (L("recovered") if res.resolved else L("not_recovered"), T["resolved"] if res.resolved else T["alert"])
                elif obs and obs["alerts"]:
                    badge = (L("working"), T["ink"])
                elif any(e["frame"] <= ss.t for e in run["events"]):
                    badge = (L("watching"), T["resolved"])
                else:
                    badge = (L("waiting_alert"), T["muted"])
                with st.container(border=True, key=k):
                    st.markdown(f'<div class="sidehead"><p class="ptitle">{title}</p><span class="badge" '
                                f'style="color:{badge[1]};border-color:{badge[1]}">{badge[0]}</span></div>',
                                unsafe_allow_html=True)
                    show_map(obs, run["events"], min(ss.t, n - 1), height=390)
        with right:
            details_panel(a, b, ss.t, s_)
        flow_log(a, b, ss.t)

    stage()


# ---------------------------------------------------------------- live
def live_tab():
    c = st.columns([1.6, 1, 1.4, 1], vertical_alignment="bottom")
    kind = c[0].selectbox(L("fault_inject"), FAULT_KINDS, format_func=fault_name, index=FAULT_KINDS.index("node_pressure"))
    targets = {"slow_query": ["db"], "node_pressure": ["node-a", "node-b"], "shifting": ["payment", "inventory"]}.get(
        kind, ["frontend", "cart", "catalog", "checkout", "payment", "inventory"])
    target = c[1].selectbox(L("where"), targets)
    opts = L("reasoners")
    reasoner = c[2].selectbox(L("reasoner"), [0, 1], format_func=lambda i: opts[i])
    if c[3].button(L("inject"), type="primary", use_container_width=True):
        ss.live = start_live(kind, target, "Ours", reasoner == 1, 0.0)
        ss.live_focus = None
    run = ss.get("live")
    if run is None:
        st.markdown(f'<div class="panel"><p class="ptitle">{L("no_live_t")}</p><p>{L("no_live")}</p></div>',
                    unsafe_allow_html=True)
        return
    if hasattr(run.get("agent"), "res"):
        run["events"] = parse_events(run["agent"].res.trace)
    human = run["human"]
    l, r = st.columns([1.9, 1])
    n = len(run["frames"])
    events = [e for e in run["events"] if e["tag"] != "REASON"]
    focus = ss.get("live_focus")
    if not run["done"] or focus is None or focus > n - 1:
        focus = max(0, n - 1)
    with l:
        obs = run["frames"][focus] if n else None
        with st.container(border=True, key="panel-live"):
            show_map(obs, run["events"], focus, height=560)
    with r:
        if human.pending is not None:
            a = human.pending
            appr = L("approve")
            title, body, yes, no = appr.get(a.name, appr["_"])
            fmt = dict(t=a.target, a=str(a))
            st.markdown(f'<div class="approval"><p class="ptitle">{L("your_call")}</p><p><b>{title.format(**fmt)}</b></p>'
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
        status = L("done") if run["done"] else (L("waiting_you") if human.pending is not None else L("working_live"))
        if not run["done"] and human.pending is None and run["model"] != "scripted reasoner":
            status += L("asking", m=run["model"])
        hint = ("Click a step to replay it on the map." if lang() == "en" else "คลิกขั้นตอนเพื่อเล่นซ้ำบนแผนผัง") if run["done"] else ""
        with st.container(border=True, key="panel-livesteps", height=624 - (236 if human.pending is not None else 0)):
            st.markdown(f'<p class="ptitle" style="margin:0">{status}</p><p class="muted" style="font-size:13px;margin:0 0 6px">'
                        f'{hint}</p>', unsafe_allow_html=True)
            for k, e in enumerate(events, 1):
                label = f"{k}  {ev_label(e).upper()}  {ev_text(e)}"
                active = run["done"] and e["frame"] == focus
                if st.button(label, key=f"lstep{k}", use_container_width=True, disabled=not run["done"],
                             type="primary" if active else "secondary"):
                    ss.live_focus = e["frame"]
                    st.rerun()
    if run.get("error"):
        st.error(L("stopped", e=run["error"]))
    if not run["done"] and human.pending is None:
        time.sleep(0.6)
        st.rerun()


# ---------------------------------------------------------------- page
css()
st.markdown(f"""<style>
.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1.5px solid {T['ink']}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 0; padding: 8px 22px; }}
.stTabs [data-baseweb="tab"] p, .stTabs button[role="tab"] p {{ font-size: 18px !important; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; }}
.stTabs [aria-selected="true"], [data-testid="stTab"][aria-selected="true"] {{ background: {T['ink']}; }}
[data-testid="stTab"] {{ padding: 8px 22px !important; border-radius: 0 !important; }}
[data-testid="stTab"] p {{ font-size: 19px !important; font-weight: 800 !important; letter-spacing: .07em; text-transform: uppercase; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {T['surface']} !important; }}
.stTabs [aria-selected="true"] p {{ color: {T['surface']} !important; }}
p.lead {{ font-size: 17px !important; line-height: 1.5; max-width: 70em; margin: 6px 0 12px; }}
.label {{ font-size: 13px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; margin: 0 0 6px; }}
.panel p.figure {{ font-size: 34px !important; line-height: 1.15 !important; font-weight: 800; color: {T['deep']} !important; margin: 0 0 4px; letter-spacing: .02em; }}
.panel p.figure .arrow {{ font-size: 15px !important; font-weight: 400; color: {T['muted']} !important; }}
p.clock {{ font-size: 20px !important; font-weight: 800; color: {T['deep']} !important; margin: 0; }}
.steps.big b {{ display: inline; margin-right: 6px; }}
div[role="radiogroup"] label p {{ font-size: 15px; }}
div[data-testid="stVerticalBlockBorderWrapper"], div[class*="st-key-panel"] {{ border: 2px solid {T['ink']} !important;
  border-radius: 0 !important; background: {T['surface']}; padding: 14px 16px !important; }}
.sidehead {{ display: flex; align-items: center; gap: 14px; min-height: 44px; margin: 2px 0 6px; }}
.sidehead .ptitle {{ line-height: 1.35 !important; }}
div[class*="st-key-lstep"] button {{ justify-content: flex-start !important; text-align: left !important;
  border: none !important; border-top: 1px dashed {T['line']} !important; padding: 10px 6px !important; min-height: 0 !important; }}
div[class*="st-key-lstep"] button p {{ font-size: 16px !important; font-weight: 600 !important; text-align: left !important; width: 100%; }}
div[class*="st-key-lstep"] button > div, div[class*="st-key-lstep"] button div[data-testid="stMarkdownContainer"] {{ justify-content: flex-start !important; text-align: left !important; width: 100%; }}
div[class*="st-key-lstep"] button[kind="primary"] {{ background: {T['tint']} !important; }}
div[class*="st-key-lstep"] button[kind="primary"] p {{ color: {T['deep']} !important; }}
.clock-s {{ margin-left: auto; font-size: 15px; font-weight: 800; color: {T['deep']} !important; }}
.sidehead .ptitle {{ margin: 0; }}
.steps.big li {{ font-size: 17px !important; font-weight: 500; line-height: 1.45; padding: 8px 0; }}
.steps.big b {{ font-size: 16px !important; }}
.steps.big .num {{ width: 26px; height: 26px; font-size: 14px; }}
.score {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; border-top: 1.5px solid {T['ink']};
  margin-top: 8px; padding-top: 10px; }}
.score p {{ margin: 0; }} .score p.big {{ font-size: 32px !important; font-weight: 800; color: {T['deep']} !important; }}
.badge {{ font-size: 14px !important; padding: 3px 10px !important; }}
.ptitle {{ font-size: 22px !important; }}
p, li, td {{ font-weight: 500; }}
span[data-testid="stIconMaterial"] {{ font-family: "Material Symbols Rounded" !important; }}
.block-container {{ padding-top: .7rem !important; padding-bottom: .4rem !important; }}
div[data-testid="stVerticalBlock"] {{ gap: .55rem; }}
button[data-variant="segmented_control"] {{ border-radius: 0 !important; border: 1.5px solid {T['ink']} !important;
  background: {T['surface']} !important; padding: 7px 30px !important; }}
button[data-variant="segmented_control"] p, button[data-variant="segmented_control"] span {{ font-size: 17px !important;
  font-weight: 800 !important; letter-spacing: .08em; text-transform: uppercase; color: {T['ink']} !important; }}
button[data-variant="segmented_control"][aria-checked="true"] {{ background: {T['ink']} !important; }}
button[data-variant="segmented_control"][aria-checked="true"] p, button[data-variant="segmented_control"][aria-checked="true"] span {{ color: {T['surface']} !important; }}
.apptitle {{ font-size: 24px !important; }}
.appsub {{ margin: 0 !important; font-size: 12px !important; letter-spacing: .1em !important; }}
.apphead {{ padding-bottom: 6px; }}
div[class*="st-key-panel"] {{ padding: 10px 14px !important; gap: .3rem !important; }}
.panel {{ margin-bottom: 0 !important; }}
.details {{ height: 448px; overflow-y: auto; }}
.det-time {{ font-size: 20px !important; font-weight: 800; color: {T['deep']} !important; margin: 0 0 8px; }}
.det-block {{ border-top: 1.5px solid {T['ink']}; padding: 8px 0 6px; }}
.det-who {{ font-size: 15px !important; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; margin: 0 0 4px; }}
.det-last {{ font-size: 16px !important; line-height: 1.45; margin: 0 0 4px; }}
.det-tag {{ color: #fff !important; font-size: 12px; font-weight: 800; letter-spacing: .05em; text-transform: uppercase; padding: 2px 7px; }}
.det-why {{ font-size: 14px !important; color: {T['muted']} !important; margin: 4px 0 0; line-height: 1.4; }}
.det-story {{ font-size: 13px !important; color: {T['muted']} !important; border-top: 1px dashed {T['line']}; padding-top: 8px; margin: 6px 0 0; }}
.flowhead {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
.flowhead .ptitle {{ margin: 0 8px 0 0; }}
.leg {{ font-size: 13px; }}
.flowbody {{ height: 170px; overflow-y: auto; margin-top: 6px; }}
.flowbody .log td {{ font-size: 14px; }}
.livesteps {{ overflow-y: auto; }}
button[role="tab"] p, button[data-baseweb="tab"] p, button[role="tab"] div {{ font-size: 18px !important;
  font-weight: 800 !important; letter-spacing: .06em; text-transform: uppercase; }}
</style>""", unsafe_allow_html=True)
ss = st.session_state
h1, h2, h3, h4 = st.columns([1.5, 1.5, 0.55, 0.55], vertical_alignment="center")
h4.segmented_control("Language", ["EN", "TH"], default="EN", key="lang", label_visibility="collapsed")
if not ss.get("lang"):
    ss.lang = "EN"
h1.markdown(f'<div class="apphead"><p class="apptitle">Reconsider before you act</p>'
            f'<p class="appsub">{L("subtitle")}</p></div>', unsafe_allow_html=True)
names = L("views")
view = h2.segmented_control("View", [0, 1, 2], default=0, format_func=lambda i: names[i],
                            label_visibility="collapsed", key="view_i")
view = 0 if view is None else view
ss.motion = h3.toggle(L("animation"), value=ss.get("motion", True), help=L("animation_help"))
if view == 0:
    replay_tab()
elif view == 1:
    results_tab()
else:
    live_tab()
