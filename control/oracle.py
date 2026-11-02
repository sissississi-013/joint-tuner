"""Numerical oracle: search PID gains per task (Nelder-Mead on a shaped penalty). Gives solvability ceiling + SFT labels."""
import sys, json, numpy as np, concurrent.futures as cf
from scipy.optimize import minimize
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from judge import load_tasks, evaluate

def penalty(x, t):
    Kp, Ki, Kd = np.exp(x)  # positive gains in log space
    for sign in (1,):
        r = evaluate(t, sign*Kp, sign*Ki, sign*Kd)
    if not r["stable"]: return 1e3
    if r.get("error"): return 1e3
    if not np.isfinite(r.get("settling_time", np.nan)): return 500.0  # never settled
    p = 0.0
    p += max(0, t["gain_margin_min"] - r["gain_margin_db"]) / 5
    p += max(0, t["phase_margin_min"] - r["phase_margin"]) / 20
    p += max(0, t["settling_time_min"] - r["settling_time"]) / t["settling_time_min"]
    p += max(0, r["settling_time"] - t["settling_time_max"]) / t["settling_time_max"]
    p += max(0, r["overshoot"] - t["overshoot_max"]) / 10
    p += max(0, r["ess"] - max(t["steadystate_error_max"], 1e-4)) * 100
    return p

from scipy.optimize import differential_evolution
def solve(t):
    res = differential_evolution(penalty, bounds=[(-9, 9), (-9, 9), (-12, 6)], args=(t,), maxiter=60, popsize=12, tol=0,
                                 seed=0, polish=False, init="sobol")
    if res.fun > 0:
        r2 = minimize(penalty, res.x, args=(t,), method="Nelder-Mead", options={"maxiter": 400})
        if r2.fun < res.fun: res = r2
    Kp, Ki, Kd = np.exp(res.x)
    r = evaluate(t, Kp, Ki, Kd)
    return {"uid": t["uid"], "category": t["category"], "Kp": Kp, "Ki": Ki, "Kd": Kd, "success": r["success"], "score": r.get("score", 0), "pen": float(res.fun)}

def solve_old(t):
    best = None
    G0 = abs(np.polyval(t["num"], 0) / np.polyval(t["den"], 0)) if np.polyval(t["den"], 0) != 0 else 1.0
    for kp0 in [0.5 / max(G0, 1e-3), 2 / max(G0, 1e-3), 1.0, 10.0]:
        for ratio in [0.3, 3.0]:
            x0 = np.log([kp0, kp0 * ratio, kp0 * 0.05])
            res = minimize(penalty, x0, args=(t,), method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-6})
            if best is None or res.fun < best.fun: best = res
            if best.fun == 0: break
        if best.fun == 0: break
    Kp, Ki, Kd = np.exp(best.x)
    r = evaluate(t, Kp, Ki, Kd)
    return {"uid": t["uid"], "category": t["category"], "Kp": Kp, "Ki": Ki, "Kd": Kd, "success": r["success"], "score": r.get("score", 0)}

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # per category
    tasks = load_tasks(); by = {}
    for t in tasks: by.setdefault(t["category"], []).append(t)
    pick = [t for c in sorted(by) for t in by[c][:n]]
    with cf.ProcessPoolExecutor(8) as ex: res = list(ex.map(solve, pick))
    json.dump(res, open(f"runs/oracle_{n}.json", "w"), indent=1)
    from collections import defaultdict
    agg = defaultdict(list)
    for r in res: agg[r["category"]].append(r["success"])
    for c, v in sorted(agg.items()): print(f"{c:40s} {sum(v)}/{len(v)}")
    print("TOTAL", sum(r["success"] for r in res), "/", len(res))
