"""Look at why LLM replies failed to parse.  python tools/inspect_log.py results/llm_pilot/llm_log.jsonl"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rbya.reasoners import _extract_json

path = sys.argv[1]
recs = [json.loads(l) for l in open(path, encoding="utf-8")]
bad = [r for r in recs if _extract_json(r["raw"]) is None]
print(f"{len(recs)} replies, {len(bad)} unparseable, mean {sum(r['seconds'] for r in recs) / max(1, len(recs)):.1f} s")
print("empty:", sum(not r["raw"].strip() for r in bad),
      "| has <think>:", sum("<think>" in r["raw"] for r in bad),
      "| <think> never closed:", sum("<think>" in r["raw"] and "</think>" not in r["raw"] for r in bad),
      "| contains '{':", sum("{" in r["raw"] for r in bad))
print("reply length (chars) of unparseable:", sorted(len(r["raw"]) for r in bad)[:20])
for r in bad[:3]:
    print("\n----- start of reply -----\n" + r["raw"][:600] + "\n----- end of reply -----\n" + r["raw"][-400:])
