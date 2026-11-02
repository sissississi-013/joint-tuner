"""Record every simulator evaluation (gains, pass/fail) for: (a) ours = 9B proposals then repair, (b) optimizer cold start.
Gives success-vs-evaluation-budget curves + per-trial traces for the demo. Usage: python control/budget_trace.py CKPT N TAG -> runs/trace_{TAG}.json"""
import os, sys, json, time, numpy as np, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from scipy.optimize import minimize
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")
try:  # River is a private training/sampling SDK; only needed when FW_MODEL is not set
    import river_client as river
    from river_client.renderers import get_renderer
except ImportError:
    river = get_renderer = None
from judge import load_tasks, evaluate, task_prompt
from propose_repair_pen import penalty
from fw import extract_json
BASE = "Qwen/Qwen3.5-9B"; CKPT, N, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3]; CAP = 600
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
MSGS = lambda t: [{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}]
FW_MODEL = os.environ.get("FW_MODEL")  # e.g. accounts/<acct>/models/pid-tuner-q36v3#accounts/<acct>/deployments/pid-tuner
if not FW_MODEL and river is None:
    sys.exit("river_client (private River SDK) is not installed. Set FW_MODEL=accounts/<acct>/models/<model> to sample from Fireworks instead.")
rend = None if FW_MODEL else get_renderer(BASE, thinking=False); stops = rend.get_stop_strings() if rend else None
class _O:
    def __init__(s, text): s.text = text
class _FW:
    def __init__(s):
        from openai import OpenAI
        s.c = OpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"])
    def sample(s, msgs, num_samples, temperature, **_):
        r = s.c.chat.completions.create(model=FW_MODEL, messages=msgs, n=num_samples, temperature=temperature, max_tokens=120,
                                        extra_body={"reasoning_effort": "none"})
        return [[_O(ch.message.content or "") for ch in r.choices]]
sess = _FW() if FW_MODEL else river.Client(api_key=os.environ["RIVER_API_KEY"]).session(project="pid-tuner", experiment="trace").__enter__()
by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
EV = [t for c in sorted(by) for t in by[c][:10]]
class Tracker:
    def __init__(s, t): s.t, s.log, s.first = t, [], None
    def ev(s, g):
        g = [float(v) for v in g]; ok = bool(evaluate(s.t, *g)["success"]); s.log.append([round(v, 6) for v in g] + [ok])
        if ok and s.first is None: s.first = len(s.log)
        return ok
    def f(s, x):
        if len(s.log) >= CAP or s.first: return 0.0 if s.first else 1e6
        g = np.exp(x); s.ev(g); return 0.0 if s.log[-1][3] else penalty(x, s.t)
def nm(tr, starts, budget):
    for x0 in starts:
        if tr.first or len(tr.log) >= CAP: break
        minimize(tr.f, x0, method="Nelder-Mead", options={"maxfev": budget, "xatol": 1e-4, "fatol": 0})
def ours(t):
    p = None if FW_MODEL else rend.build_text_prompt([{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}])
    for a in range(5):
        try: outs = sess.sample(MSGS(t) if FW_MODEL else p, base_model=BASE, checkpoint=CKPT, num_samples=N, max_tokens=60, temperature=1.0, stop=stops)[0]; break
        except Exception: time.sleep(3 * (a + 1))
    else: outs = []
    cands = []
    for o in outs:
        j = extract_json(o.text) or {}
        try: cands.append([float(j["Kp"]), float(j["Ki"]), float(j["Kd"])])
        except Exception: pass
    tr = Tracker(t)
    for g in cands:
        if tr.ev(g): return tr
    starts = sorted(cands, key=lambda g: penalty(np.log(np.maximum(np.abs(g), 1e-6)), t))[:3]
    if "unstable" in t["category"]: starts += [[g[0], g[1], g[0] * k] for g in starts[:2] for k in (0.3, 1.0, 3.0)]
    nm(tr, [np.log(np.maximum(np.abs(g), 1e-6)) for g in starts], 150 * (3 if "unstable" in t["category"] else 1))
    return tr
def cold(t):
    tr = Tracker(t); rng = np.random.default_rng(0)
    g0 = abs(np.polyval(t["num"], 0) / np.polyval(t["den"], 0)) if abs(np.polyval(t["den"], 0)) > 1e-9 else 1.0
    starts = [np.log([1 / max(g0, 1e-3), 1 / max(g0, 1e-3), 0.01 / max(g0, 1e-3)])] + [rng.uniform([-4, -4, -8], [6, 8, 2]) for _ in range(9)]
    nm(tr, starts, 150); return tr
def both(t):
    a, b = ours(t), cold(t)
    return {"uid": t["uid"], "category": t["category"], "ours_first": a.first, "cold_first": b.first, "ours_log": a.log, "cold_log": b.log}
with cf.ThreadPoolExecutor(6) as ex: rows = list(ex.map(both, EV))
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 600]
curve = {k: [sum(1 for r in rows if r[f"{k}_first"] and r[f"{k}_first"] <= B) for B in Bs] for k in ("ours", "cold")}
print("budget:", Bs); print("ours  :", curve["ours"]); print("cold  :", curve["cold"])
json.dump({"budgets": Bs, "curve": curve, "rows": rows}, open(f"runs/trace_{TAG}.json", "w"))
