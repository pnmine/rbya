"""Figures for the paper from results/*/summary.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.size": 9})
ARMS = ["B1-ReAct", "B2-ActRollback", "Ours-noCompile", "Ours"]
LABEL = {"B1-ReAct": "B1 ReAct", "B2-ActRollback": "B2 act+rollback", "Ours-noCompile": "Ours w/o compile", "Ours": "Ours"}
COL = {"B1-ReAct": "#8c8c8c", "B2-ActRollback": "#5b6168", "Ours-noCompile": "#e39461", "Ours": "#9c4318"}

main = json.load(open("results/scripted/summary.json"))["summary"]
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
for ax, key, title in zip(axes, ["unnecessary_per_incident", "collateral_ticks_total", "mttr_min"],
                          ["Unneeded actions / incident", "Collateral ticks (total)", "MTTR (min)"]):
    vals = [main[a][key] for a in ARMS]
    ax.bar(range(4), vals, color=[COL[a] for a in ARMS])
    ax.set_xticks(range(4), [LABEL[a] for a in ARMS], rotation=35, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("figures/main_results.pdf")
fig.savefig("figures/main_results.png", dpi=200)

slips = [0.0, 0.25, 0.5]
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
for ax, key, title in zip(axes, ["unnecessary_per_incident", "llm_calls_per_incident"],
                          ["Unneeded actions / incident", "Reasoner calls / incident"]):
    for a in ARMS:
        ys = [json.load(open(f"results/slip_{s}/summary.json"))["summary"][a][key] for s in slips]
        ax.plot(slips, ys, marker="o", label=LABEL[a], color=COL[a])
    ax.set_xlabel("reasoner slip probability")
    ax.set_title(title, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig("figures/slip_sensitivity.pdf")
fig.savefig("figures/slip_sensitivity.png", dpi=200)
print("figures written")
