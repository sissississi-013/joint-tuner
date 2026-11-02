"""Zero-shot PID baseline. Usage: python control/probe_llm.py PROVIDER MODEL PER_CAT
PROVIDER: river | fireworks"""
import sys, os, json, time, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
from judge import load_tasks, evaluate, task_prompt
from fw import extract_json
load_dotenv(Path(__file__).parent.parent / ".env")
prov, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
THINK = (len(sys.argv) > 4 and sys.argv[4] == "think")
cli = OpenAI(api_key=os.environ["RIVER_API_KEY"], base_url="https://api.river.ai/v1") if prov == "river" else \
      OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1")
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
OFF = int(os.environ.get("OFFSET", "0")); pick = [t for c in sorted(by) for t in by[c][OFF:OFF + n]]

def run(t):
    t0 = time.perf_counter()
    for attempt in range(6):
      try:
        r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}],
                                        max_tokens=(16000 if THINK else 1000), temperature=0.0,
                                        **({} if THINK else {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}))
        txt = r.choices[0].message.content or ""; ctok = r.usage.completion_tokens if r.usage else None; err = None
        break
      except Exception as e:
        txt, ctok, err = "", None, str(e)[:200]
        if "429" in err: time.sleep(5 * (attempt + 1)); continue
        break
    dt = time.perf_counter() - t0
    j = extract_json(txt) or {}
    ev = evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd")) if j else {"stable": False, "success": False}
    return {"uid": t["uid"], "category": t["category"], "latency_s": dt, "completion_tokens": ctok, "error": err,
            "parsed": bool(j), "gains": j, "success": ev["success"], "stable": ev["stable"], "score": float(ev.get("score", 0)), "checks": {k: bool(v) for k, v in ev.get("checks", {}).items()}}

with cf.ThreadPoolExecutor(4) as ex: res = list(ex.map(run, pick))
tag = f"{prov}_{model.replace('/', '_')}_{n}{'_think' if THINK else ''}{'_off'+str(OFF) if OFF else ''}"
json.dump(res, open(f"runs/pid_{tag}.json", "w"), indent=1)
agg = defaultdict(list)
for r in res: agg[r["category"]].append(r["success"])
for c, v in sorted(agg.items()): print(f"{c:40s} {sum(v)}/{len(v)}")
ok = [r for r in res if not r["error"]]; lat = sorted(r["latency_s"] for r in ok)
print(f"MODEL {model}: success {sum(r['success'] for r in res)}/{len(res)} | stable {sum(r['stable'] for r in res)} | parsed {sum(r['parsed'] for r in res)} | errors {len(res)-len(ok)}")
if lat: print(f"p50 {lat[len(lat)//2]:.1f}s  mean ctok {sum(r['completion_tokens'] or 0 for r in ok)/len(ok):.0f}  mean partial score {sum(r['score'] for r in res)/len(res):.2f}")
errs = [r['error'] for r in res if r['error']]
if errs: print("first error:", errs[0])
