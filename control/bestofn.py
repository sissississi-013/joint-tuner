"""Verifier-in-the-loop: sample N candidates from a River checkpoint, judge each with the simulator, pick a passing one.
Reports success@k for k in 1,2,4,8 (prefix of the N samples) on ControlEval (10/category). Usage: python control/bestofn.py BASE CKPT N TEMP TAG"""
import os, sys, json, time, concurrent.futures as cf
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")
try:
    import river_client as river
    from river_client.renderers import get_renderer
except ImportError:
    sys.exit("This script needs river_client, a private River SDK that is not public. Its logged outputs are in runs/.")
from judge import load_tasks, evaluate, task_prompt
from fw import extract_json
BASE, CKPT, N, TEMP, TAG = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
PER = int(os.environ.get("PER", "10"))
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
rend = get_renderer(BASE, thinking=False); stops = rend.get_stop_strings()
by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
ev = [t for c in sorted(by) for t in by[c][:PER]]
client = river.Client(api_key=os.environ["RIVER_API_KEY"])
with client.session(project="pid-tuner", experiment=f"bon-{TAG}") as s:
    def one(t):
        p = rend.build_text_prompt([{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}])
        t0 = time.time(); outs = []
        for a in range(5):
            try: outs = s.sample(p, base_model=BASE, checkpoint=(None if CKPT=="none" else CKPT), num_samples=N, max_tokens=60, temperature=TEMP, stop=stops)[0]; break
            except Exception as e: time.sleep(3 * (a + 1))
        lat = time.time() - t0; tj = time.time(); oks = []
        for o in outs:
            j = extract_json(o.text) or {}
            oks.append(bool(evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd"))["success"]) if j else False)
        return {"uid": t["uid"], "category": t["category"], "oks": oks, "sample_s": lat, "judge_s": time.time() - tj}
    with cf.ThreadPoolExecutor(4) as ex: res = list(ex.map(one, ev))
json.dump(res, open(f"runs/bon_{TAG}.json", "w"), indent=1)
for k in [1, 2, 4, 8, 16, 32]:
    if k > N: break
    s_ = sum(any(r["oks"][:k]) for r in res); print(f"success@{k}: {s_}/{len(res)} = {s_/len(res):.0%}")
agg = defaultdict(list)
for r in res: agg[r["category"]].append(any(r["oks"]))
print("by cat @N:", " ".join(f"{c.replace('first_order','fo').replace('second_order','so')}={sum(v)}/{len(v)}" for c, v in sorted(agg.items())))
print(f"mean sample latency {sum(r['sample_s'] for r in res)/len(res):.2f}s, judge {sum(r['judge_s'] for r in res)/len(res)*1000:.0f}ms for {N} candidates")
