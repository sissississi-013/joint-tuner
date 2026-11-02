"""Ablation: same repair optimizer, but cold-started (no LLM) with a fixed simulator budget. Answers 'why not just run an optimizer?'.
Usage: python control/coldstart_ablation.py BUDGET_PER_START N_STARTS -> runs/coldstart_{B}x{S}.json"""
import sys, json, time, numpy as np, concurrent.futures as cf
from collections import defaultdict
from scipy.optimize import minimize
sys.path.insert(0, "control")
from judge import load_tasks, evaluate
import propose_repair_pen as P
B, S = int(sys.argv[1]), int(sys.argv[2])
by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
EV = [t for c in sorted(by) for t in by[c][:10]]
def one(t):
    t0 = time.time(); calls = [0]; rng = np.random.default_rng(0)
    def f(x): calls[0] += 1; return P.penalty(x, t)
    g0 = abs(np.polyval(t["num"], 0) / np.polyval(t["den"], 0)) if abs(np.polyval(t["den"], 0)) > 1e-9 else 1.0
    starts = [np.log([1 / max(g0, 1e-3), 1 / max(g0, 1e-3), 0.01 / max(g0, 1e-3)])] + [rng.uniform([-4, -4, -8], [6, 8, 2]) for _ in range(S - 1)]
    ok = False
    for x0 in starts:
        r = minimize(f, x0, method="Nelder-Mead", options={"maxfev": B, "xatol": 1e-4, "fatol": 0})
        if evaluate(t, *np.exp(r.x))["success"]: ok = True; break
    return {"uid": t["uid"], "category": t["category"], "success": ok, "sim_calls": calls[0], "t": time.time() - t0}
with cf.ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, EV))
c = defaultdict(int)
for r in rows: c[r["category"]] += r["success"]
ts = sorted(r["t"] for r in rows)
print(f"COLD-START NM budget {B}x{S}: {sum(r['success'] for r in rows)}/100 | mean sim calls {np.mean([r['sim_calls'] for r in rows]):.0f} | p50 {ts[50]:.1f}s |", " ".join(f"{k.replace('second_order','so').replace('first_order','fo')}={v}" for k, v in sorted(c.items())))
json.dump(rows, open(f"runs/coldstart_{B}x{S}.json", "w"), indent=1)
