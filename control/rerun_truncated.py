"""Re-run Kimi K3 thinking on tasks truncated at 16k tokens, with a 100k budget. Writes runs/pid_fireworks_kimi-k3_think_full100.json (all 100)."""
import os, sys, json, time, concurrent.futures as cf
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")
from openai import OpenAI
from judge import load_tasks, evaluate, task_prompt
from fw import extract_json
cli = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1", timeout=3600)
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
T = {t["uid"]: t for t in load_tasks()}
rows = json.load(open("runs/pid_fireworks_kimi-k3_5_think.json")) + json.load(open("runs/pid_fireworks_kimi-k3_5_think_off5.json"))
todo = [r for r in rows if (r.get("completion_tokens") or 0) >= 15990]
print(len(todo), "truncated tasks to re-run", flush=True)
def run(r):
    t = T[r["uid"]]; t0 = time.perf_counter()
    for a in range(6):
        try:
            x = cli.chat.completions.create(model="accounts/fireworks/models/kimi-k3", messages=[{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}], max_tokens=100000, temperature=0.0)
            txt = x.choices[0].message.content or ""; ctok = x.usage.completion_tokens; err = None; break
        except Exception as e:
            txt, ctok, err = "", None, str(e)[:200]; time.sleep(10 * (a + 1))
    j = extract_json(txt) or {}
    ev = evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd")) if j else {"stable": False, "success": False}
    out = {**r, "latency_s": time.perf_counter() - t0, "completion_tokens": ctok, "error": err, "parsed": bool(j), "gains": j, "success": bool(ev["success"]), "stable": bool(ev.get("stable")),
           "checks": {k: bool(v) for k, v in ev.get("checks", {}).items()}, "rerun_100k": True}
    print(r["uid"], "ok" if out["success"] else "fail", ctok, flush=True); return out
with cf.ThreadPoolExecutor(12) as ex: new = {o["uid"]: o for o in ex.map(run, todo)}
full = [new.get(r["uid"], r) for r in rows]
json.dump(full, open("runs/pid_fireworks_kimi-k3_think_full100.json", "w"), indent=1)
ct = [r["completion_tokens"] for r in full if r.get("completion_tokens")]; lat = sorted(r["latency_s"] for r in full)
print(f"KIMI THINK FULL: success {sum(r['success'] for r in full)}/100 | still>=99k {sum(1 for c in ct if c>=99000)} | p50 latency {lat[50]:.0f}s | mean out tok {sum(ct)/len(ct):.0f}", flush=True)
