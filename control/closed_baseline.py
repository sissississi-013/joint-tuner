"""Closed frontier baselines on the same ControlEval tasks + same prompt + same judge.
Usage: python control/closed_baseline.py openai|anthropic MODEL PER_CAT  -> runs/pid_closed_{model}_{n}.json"""
import os, sys, json, time, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
from judge import load_tasks, evaluate, task_prompt
from fw import extract_json
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
prov, model, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
pick = [t for c in sorted(by) for t in by[c][:n]][:int(os.environ.get("LIMIT", "100000"))]
if prov == "openai":
    from openai import OpenAI; cli = OpenAI()
    def call(t):
        r = cli.chat.completions.create(model=model, messages=[{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}], max_completion_tokens=32000)
        u = r.usage; return r.choices[0].message.content or "", u.prompt_tokens, u.completion_tokens
else:
    import anthropic; cli = anthropic.Anthropic()
    def call(t):
        with cli.messages.stream(model=model, max_tokens=32000, system=SYS, messages=[{"role": "user", "content": task_prompt(t)}]) as st: r = st.get_final_message()
        return "".join(b.text for b in r.content if b.type == "text"), r.usage.input_tokens, r.usage.output_tokens
def run(t):
    for a in range(6):
        try:
            t0 = time.time(); txt, pt, ct_ = call(t); lat = time.time() - t0; break
        except Exception as e:
            err = str(e)[:200]; time.sleep(5 * (a + 1))
    else:
        return {"uid": t["uid"], "category": t["category"], "error": err, "success": False}
    j = extract_json(txt) or {}
    ev = evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd")) if j else {"success": False}
    return {"uid": t["uid"], "category": t["category"], "latency_s": lat, "prompt_tokens": pt, "completion_tokens": ct_, "gains": j, "success": bool(ev.get("success")), "checks": {k: bool(v) for k, v in ev.get("checks", {}).items()}}
with cf.ThreadPoolExecutor(4) as ex: res = list(ex.map(run, pick))
ok = sum(r["success"] for r in res); lats = sorted(r["latency_s"] for r in res if "latency_s" in r)
print(f"MODEL {model}: success {ok}/{len(res)} | errors {sum('error' in r for r in res)} | p50 {lats[len(lats)//2] if lats else 0:.1f}s | mean out tok {sum(r.get('completion_tokens',0) for r in res)/max(len(lats),1):.0f}")
json.dump(res, open(f"runs/pid_closed_{model}_{n}.json", "w"), indent=1)
