"""Offline re-run of the Joint Tuner REPAIR stage (no LLM calls).
Replays the logged LLM proposals from runs/trace_v3.json, then runs a new repair schedule:
  1. LLM proposals (replayed, each counted as an evaluation)
  2. [opt] cold's DC-gain anchor point (1 eval)
  3. Nelder-Mead from top-3 LLM seeds (+ derivative-heavy seeds for unstable plants), with a TOTAL budget cap
  4. [opt] scaled LLM seeds (x3, x10 of all gains) for unstable plants
  5. fallback: exactly cold()'s restart schedule (DC-gain start + 9 rng(0) uniform starts, 150 fev each)
Hard CAP=600 evaluations including proposals. Usage: python control/repair_v2.py [VARIANT] -> runs/trace_v3_repair2.json"""
import sys, json, time, numpy as np, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).parent))
from judge import load_tasks, evaluate
from propose_repair_pen import penalty
ROOT = Path(__file__).parent.parent; CAP = 600
Bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 600]
# variant -> (anchor, seed_cap_stable, seed_cap_unstable, scaled_seeds, cold_fallback)
VARIANTS = {
    "v1":      (False, None, None, False, False),  # original repair (sanity replay)
    "a":       (False, None, None, False, True),   # (a) cold fallback after seeds exhausted
    "a300":    (False, 300, 300, False, True),     # (a) + cap seed-repair at 300 evals so fallback actually runs
    "ab300":   (False, 300, 300, True, True),      # + (b) scaled seeds for unstable
    "anc_ab300": (True, 300, 300, True, True),     # + cold anchor probe right after proposals
    "anc_a300": (True, 300, 300, False, True),
}

class Tracker:
    def __init__(s, t): s.t, s.log, s.first, s.lim = t, [], None, CAP
    def ev(s, g):
        g = [float(v) for v in g]; ok = bool(evaluate(s.t, *g)["success"]); s.log.append([round(v, 6) for v in g] + [ok])
        if ok and s.first is None: s.first = len(s.log)
        return ok
    def done(s): return bool(s.first) or len(s.log) >= min(s.lim, CAP)
    def f(s, x):
        if s.done(): return 0.0 if s.first else 1e6
        s.ev(np.exp(x)); return 0.0 if s.log[-1][3] else penalty(x, s.t)

def nm(tr, starts, budget):
    for x0 in starts:
        if tr.done(): break
        minimize(tr.f, x0, method="Nelder-Mead", options={"maxfev": budget, "xatol": 1e-4, "fatol": 0})

L = lambda g: np.log(np.maximum(np.abs(np.asarray(g, float)), 1e-6))

def split_proposals(t, log):
    """Find how many leading log entries were LLM proposals: the repair's first eval is exp(x0) of the
    lowest-penalty proposal (NM evaluates x0 first). If a proposal succeeded, all entries are proposals."""
    if log and log[-1][3] and len(log) <= 8 and all(not e[3] for e in log[:-1]): return len(log)
    for i in range(1, min(8, len(log) - 1) + 1):
        c = [e[:3] for e in log[:i]]
        best = sorted(c, key=lambda g: penalty(L(g), t))[0]
        if np.allclose(np.round(np.exp(L(best)), 6), log[i][:3], rtol=1e-5, atol=1e-6): return i
    return min(8, len(log))

def cold_starts(t):
    rng = np.random.default_rng(0)
    g0 = abs(np.polyval(t["num"], 0) / np.polyval(t["den"], 0)) if abs(np.polyval(t["den"], 0)) > 1e-9 else 1.0
    return [np.log([1 / max(g0, 1e-3), 1 / max(g0, 1e-3), 0.01 / max(g0, 1e-3)])] + [rng.uniform([-4, -4, -8], [6, 8, 2]) for _ in range(9)]

def repair(t, cands, variant):
    anchor, cap_s, cap_u, scaled, fallback = VARIANTS[variant]
    unst = "unstable" in t["category"]; tr = Tracker(t)
    for g in cands:
        if tr.ev(g): return tr
    cs = cold_starts(t)
    if anchor and tr.ev(np.exp(cs[0])): return tr
    starts = sorted(cands, key=lambda g: penalty(L(g), t))[:3]
    if unst: starts += [[g[0], g[1], g[0] * k] for g in starts[:2] for k in (0.3, 1.0, 3.0)]
    cap = cap_u if unst else cap_s
    tr.lim = CAP if cap is None else min(CAP, len(tr.log) + cap)
    nm(tr, [L(g) for g in starts], 150 * (3 if unst else 1))
    tr.lim = CAP
    if scaled and unst and not tr.done():
        top = sorted(cands, key=lambda g: penalty(L(g), t))[:2]
        tr.lim = min(CAP, len(tr.log) + 120)
        nm(tr, [L(g) + np.log(k) for k in (3.0, 10.0) for g in top], 60)
        tr.lim = CAP
    if fallback and not tr.done():
        nm(tr, cs[1:] if anchor else cs, 150)
    return tr

def run(variant, rows, T):
    def one(r):
        t = T[r["uid"]]; n = split_proposals(t, r["ours_log"])
        if r["ours_first"] and r["ours_first"] <= n:  # a proposal already passed: repair never runs, replay as-is
            return dict(r, n_proposals=n)
        tr = repair(t, [e[:3] for e in r["ours_log"][:n]], variant)
        return {"uid": r["uid"], "category": r["category"], "ours_first": tr.first, "cold_first": r["cold_first"],
                "ours_log": tr.log, "cold_log": r["cold_log"], "n_proposals": n}
    with cf.ThreadPoolExecutor(8) as ex: return list(ex.map(one, rows))

curve = lambda rows, k: [sum(1 for r in rows if r[f"{k}_first"] and r[f"{k}_first"] <= B) for B in Bs]

if __name__ == "__main__":
    T = {t["uid"]: t for t in load_tasks()}
    src = json.load(open(ROOT / "runs/trace_v3.json")); rows = src["rows"]
    variants = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    out_variant = sys.argv[2] if len(sys.argv) > 2 else None
    res = {}
    for v in variants:
        t0 = time.time(); res[v] = run(v, rows, T)
        print(f"{v:10s} {curve(res[v], 'ours')}  ({time.time() - t0:.0f}s)", flush=True)
    print("old ours  ", src["curve"]["ours"]); print("cold      ", src["curve"]["cold"])
    cats = sorted({r["category"] for r in rows})
    for v in variants:
        g = {c: (sum(1 for r in res[v] if r["category"] == c and r["ours_first"]), sum(1 for r in rows if r["category"] == c and r["ours_first"])) for c in cats}
        print(v, {c[:22]: f"{a[1]}->{a[0]}" for c, a in g.items() if a[0] != a[1]})
    json.dump({v: {"curve": curve(res[v], "ours"), "first": {r["uid"]: r["ours_first"] for r in res[v]}} for v in variants},
              open(ROOT / "runs/repair_v2_variants.json", "w"))
    if out_variant:
        rr = res[out_variant]
        json.dump({"budgets": Bs, "variant": out_variant, "curve": {"ours": curve(rr, "ours"), "cold": curve(rr, "cold")}, "rows": rr},
                  open(ROOT / "runs/trace_v3_repair2.json", "w"))
        print("wrote runs/trace_v3_repair2.json with variant", out_variant)
