# Reconsider Before You Act

A self-healing agent for microservices that sorts candidate fixes by risk, re-checks the symptom before acting, asks a human for high-risk actions, and keeps fixes that worked for next time. CPE 633 Cognitive Computing term project, KMUTT.

The study runs in a small discrete-time simulator of a seven-service shop (one tick = 30 s), so every incident has a known root cause and a known correct fix.

## Layout

```
rbya/
  sim.py         simulator: services, dependencies, nodes, 8 fault types, metrics/logs/traces
  actions.py     action catalog and risk tiers (low / medium / high)
  telemetry.py   evidence rendering for the reasoner, alert signatures, fault keys
  reasoners.py   ScriptedReasoner (rule-based LLM stand-in) and LLMReasoner (any OpenAI-compatible API)
  agents.py      B1 ReAct, B2 act-then-rollback, Ours (+ ablation switches)
  memory.py      procedure store (knowledge compilation)
  human.py       simulated on-call engineer (accuracy, response delay)
run_experiments.py   runs all arms on the same incident sequences
make_figures.py      paper figures from results/*/summary.json
demo.py              narrated single incident for the live demo
tests/               unit tests
```

## Requirements

Python 3.10 or newer. The core uses only the standard library. `make_figures.py` needs matplotlib.

```
pip install matplotlib
python -m unittest discover -s tests
```

## Reproduce the results

```
python run_experiments.py --episodes 30 --out results/scripted --save-traces
python run_experiments.py --episodes 30 --out results/ablation \
    --arms B1-ReAct,Abl-tiersOnly,Abl-noReconsider,Abl-noHold,Abl-noTiers,Ours
for s in 0.0 0.25 0.5; do python run_experiments.py --episodes 30 --slip $s --out results/slip_$s; done
python make_figures.py
```

Each episode is 16 incidents (8 distinct faults, each seen twice) so the procedure store can be tested on repeats. All arms see the same incident sequence.

## Demo

```
python demo.py --fault shifting --target payment          # our agent
python demo.py --fault slow_query --agent b1              # plain ReAct for comparison
python demo.py --fault node_pressure --show-evidence      # print the telemetry the agent saw
```

## Using a real LLM

`LLMReasoner` talks to any OpenAI-compatible chat endpoint. Set three environment variables:

```
export RBYA_LLM_BASE_URL=http://localhost:11434/v1     # Ollama, vLLM, or another compatible server
export RBYA_LLM_MODEL=qwen2.5:7b-instruct
export RBYA_LLM_API_KEY=none                            # or a real key
python run_experiments.py --reasoner llm --episodes 3 --out results/llm --save-traces
python demo.py --reasoner llm --fault shifting
```

## Metrics

resolution rate, MTTR (minutes from first alert to two clean ticks), unneeded actions (executed actions that did not remove the fault), actions undone, plans abandoned at the re-check, escalations to the human, collateral ticks (outage caused by the agent's own actions), reasoner calls per incident, procedure-store hits.

## Limits

The environment is a simulator, not a production cluster, and the default reasoner is scripted. Results show how the control structure behaves; they are not a claim about real-world MTTR.

## Running on the CPE AI Gateway (Windows)

After `setx RBYA_LLM_BASE_URL https://gateway.ai.cpe.kmutt.ac.th/v1`, `setx RBYA_LLM_MODEL muse-glimmer-30b`
and `setx RBYA_LLM_API_KEY ...`, open a new PowerShell window so the variables are picked up, then from the repo folder:

```
python demo.py --reasoner llm --fault shifting
python run_experiments.py --reasoner llm --episodes 1 --out results/llm_pilot --save-traces
```

- `--model qwen3.8-27b` overrides the model for one run.
- Runs resume: start the same command again and finished (episode, arm) pairs are skipped.
- Every prompt and reply goes to `<out>/llm_log.jsonl`. Identical prompts are answered from `<out>/llm_cache.jsonl`
  (`--no-cache` turns that off).
- TLS errors: `pip install truststore` (uses the Windows certificate store) or point `RBYA_LLM_CA_BUNDLE` at a PEM file.
  `RBYA_LLM_INSECURE=1` turns verification off and is a last resort only.
- `RBYA_LLM_MAX_TOKENS` (default 1024) and `RBYA_LLM_TIMEOUT` (default 180 s) control each request.

## Dashboard

```
pip install streamlit
streamlit run dashboard.py
```

- **Recorded run** replays an incident from `results/llm_*` with the model's recorded replies (instant, no API calls).
  Pick the experiment, episode, incident and agent, then press Load incident; the right column compares the same
  incident handled by B1. A link like `http://localhost:8501/?load=llm_noisy,0,11,Ours` opens one directly.
- **Live run** injects a fault now. High-risk actions (`drain_node`, `restart_database`) stop and wait for you to
  approve or reject them in the page.

The look follows the RBYA Blueprint design system (navy line work, Iosevka, square panels).
