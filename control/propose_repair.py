"""Propose -> simulate -> repair. LLM samples N candidates; if none passes, Nelder-Mead polish (log-gain space) warm-started
from the top-K candidates by spec-violation penalty, with a fixed simulator-call budget.
Usage: python control/propose_repair.py BASE CKPT N TEMP TAG   (env PER=10, TOPK=3, BUDGET=150) -> runs/repair_{TAG}.json"""
import os, sys, json, time, numpy as np, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from scipy.optimize import minimize
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")
import river_client as river
from river_client.renderers import get_renderer
from judge import load_tasks, evaluate, task_prompt
from oracle import penalty as _pen
import control as ct
from judge import plant, TF_FILTER
_S = ct.tf("s")
def penalty(x, t):
    v = _pen(x, t)
    if v < 1e3: return v
    try:  # smooth stability term: push closed-loop poles left
        Kp, Ki, Kd = np.exp(x); L = plant(t) * (Kp + Ki / _S + Kd * _S / (TF_FILTER * _S + 1))
        m = float(np.max(np.real(ct.poles(ct.feedback(L, 1)))))
        return 1e3 + 100 * np.log1p(max(m, 0) + 1e-3)
    except Exception:
        return 2e3
from fw import extract_json
BASE, CKPT, N, TEMP, TAG = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
PER, TOPK, BUDGET = int(os.environ.get("PER", "10")), int(os.environ.get("TOPK", "3")), int(os.environ.get("BUDGET", "150"))
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
rend = get_renderer(BASE, thinking=False); stops = rend.get_stop_strings()
sess = river.Client(api_key=os.environ["RIVER_API_KEY"]).session(project="pid-tuner", experiment="repair").__enter__()
by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
EV = [t for c in sorted(by) for t in by[c][:PER] if not os.environ.get('CATS') or any(k in c for k in os.environ['CATS'].split(','))]
def gains(txt):
    j = extract_json(txt) or {}
    try: return [float(j["Kp"]), float(j["Ki"]), float(j["Kd"])]
    except Exception: return None
def one(t):
    p = rend.build_text_prompt([{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}])
    t0 = time.time()
    for a in range(5):
        try: outs = sess.sample(p, base_model=BASE, checkpoint=CKPT, num_samples=N, max_tokens=60, temperature=TEMP, stop=stops)[0]; break
        except Exception: time.sleep(3 * (a + 1))
    else: outs = []
    t_llm = time.time() - t0
    cands = [g for g in (gains(o.text) for o in outs) if g]
    oks = [bool(evaluate(t, *g)["success"]) for g in cands]
    row = {"uid": t["uid"], "category": t["category"], "n_cands": len(cands), "sample_ok": any(oks), "t_llm": t_llm, "sim_calls": len(cands), "repair_ok": False, "t_repair": 0.0}
    if any(oks): row["final"] = cands[oks.index(True)]; return row
    t1 = time.time(); calls = [0]
    def f(x): calls[0] += 1; return penalty(x, t)
    starts = sorted(cands, key=lambda g: f(np.log(np.maximum(np.abs(g), 1e-6))))[:TOPK]
    if "unstable" in t["category"]:  # derivative-heavy seeds: unstable plants need phase lead
        starts = starts + [[g[0], g[1], g[0] * k] for g in starts[:2] for k in (0.3, 1.0, 3.0)]
    for g in starts:
        x0 = np.log(np.maximum(np.abs(g), 1e-6)); x0[2] = max(x0[2], np.log(max(abs(g[0]), 1e-6)) - 6)  # give Kd room if 0
        r = minimize(f, x0, method="Nelder-Mead", options={"maxfev": BUDGET * (3 if "unstable" in t["category"] else 1), "xatol": 1e-4, "fatol": 0})
        Kp, Ki, Kd = np.exp(r.x)
        if evaluate(t, Kp, Ki, Kd)["success"]: row.update(repair_ok=True, final=[Kp, Ki, Kd]); break
    row["t_repair"] = time.time() - t1; row["sim_calls"] += calls[0]
    return row
with cf.ThreadPoolExecutor(4) as ex: rows = list(ex.map(one, EV))
s = sum(r["sample_ok"] for r in rows); rp = sum(r["repair_ok"] for r in rows)
cat = defaultdict(lambda: [0, 0])
for r in rows: cat[r["category"]][0] += r["sample_ok"]; cat[r["category"]][1] += r["sample_ok"] or r["repair_ok"]
lat = sorted(r["t_llm"] + r["t_repair"] for r in rows)
print(f"[{TAG}] sample@{N} {s}/{len(rows)} -> +repair {s+rp}/{len(rows)} | p50 total {lat[len(lat)//2]:.1f}s p90 {lat[int(len(lat)*.9)]:.1f}s | mean sim calls {np.mean([r['sim_calls'] for r in rows]):.0f}")
print("by cat (sample -> +repair):", " ".join(f"{c.replace('second_order','so').replace('first_order','fo')}={a}->{b}" for c, (a, b) in sorted(cat.items())))
json.dump(rows, open(f"runs/repair_{TAG}.json", "w"), indent=1, default=float)
