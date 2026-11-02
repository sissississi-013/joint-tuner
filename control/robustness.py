"""Robustness of final passing PID gains under +/-20% plant-parameter perturbation.
Ours (Joint Tuner) vs Optimizer only (cold start) vs GPT-6.1 Terra. CPU only, no network.
Each task gets the SAME 50 perturbed plants for every method (common random numbers, seeded by uid),
so the paired comparison isolates the gains."""
import sys, os, json, time, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from judge import load_tasks, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_PERT, LO, HI, N_BOOT = 50, 0.8, 1.2, 10000

def J(x):
    return json.loads(x) if isinstance(x, str) else x

def lit(x):
    import ast
    return ast.literal_eval(x) if isinstance(x, str) else x

def final_gains():
    tr = json.load(open(f"{ROOT}/runs/trace_v3.json"))["rows"]
    ours, cold = {}, {}
    for r in tr:
        for key, log, dst in (("ours_first", "ours_log", ours), ("cold_first", "cold_log", cold)):
            f = r.get(key)
            if f not in (None, "", "None"):
                f = int(f); dst[r["uid"]] = [float(v) for v in J(r[log])[f - 1][:3]]
    gpt = {}
    for r in J(json.load(open(f"{ROOT}/runs/pid_closed_gpt-6.1-terra_10.json"))):
        if str(r.get("success")) == "True":
            g = lit(r["gains"]); gpt[r["uid"]] = [float(g["Kp"]), float(g["Ki"]), float(g["Kd"])]
    return {"ours": ours, "cold": cold, "gpt": gpt}

def perturbed_plants(t):
    rng = np.random.default_rng(zlib.crc32(t["uid"].encode()))
    out = []
    for _ in range(N_PERT):
        p = dict(t)
        p["num"] = [c * rng.uniform(LO, HI) for c in t["num"]]
        p["den"] = [t["den"][0]] + [c * rng.uniform(LO, HI) for c in t["den"][1:]]
        if t.get("tau"): p["tau"] = t["tau"] * rng.uniform(LO, HI)
        out.append(p)
    return out

def run_one(args):
    method, t, g, plants = args
    nom = evaluate(t, *g)
    res = [evaluate(p, *g) for p in plants]
    return method, t["uid"], {
        "category": t["category"], "gains": g, "nominal_success": bool(nom["success"]),
        "stable_frac": float(np.mean([r["stable"] for r in res])),
        "spec_frac": float(np.mean([r["success"] for r in res])),
    }

def boot_ci(d, rng):
    d = np.asarray(d); idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(np.mean(m <= 0))

def main():
    t0 = time.time()
    tasks = {t["uid"]: t for t in load_tasks()}
    G = final_gains()
    plants = {u: perturbed_plants(tasks[u]) for u in set().union(*G.values())}
    jobs = [(m, tasks[u], g, plants[u]) for m, d in G.items() for u, g in d.items()]
    res = {m: {} for m in G}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for m, u, r in ex.map(run_one, jobs):
            res[m][u] = r
    summary = {}
    for m, d in res.items():
        cats = sorted({r["category"] for r in d.values()})
        summary[m] = {
            "n_tasks": len(d), "nominal_reverified": sum(r["nominal_success"] for r in d.values()),
            "stable_mean": float(np.mean([r["stable_frac"] for r in d.values()])),
            "spec_mean": float(np.mean([r["spec_frac"] for r in d.values()])),
            "per_category": {c: {"n": sum(r["category"] == c for r in d.values()),
                                 "stable_mean": float(np.mean([r["stable_frac"] for r in d.values() if r["category"] == c])),
                                 "spec_mean": float(np.mean([r["spec_frac"] for r in d.values() if r["category"] == c]))} for c in cats}}
    common = sorted(set(res["ours"]) & set(res["cold"]) & set(res["gpt"]))
    rng = np.random.default_rng(0)
    paired = {"n_common": len(common),
              "mean": {m: {k: float(np.mean([res[m][u][k] for u in common])) for k in ("stable_frac", "spec_frac")} for m in res}}
    for other in ("cold", "gpt"):
        for k in ("stable_frac", "spec_frac"):
            mean, lo, hi, p = boot_ci([res["ours"][u][k] - res[other][u][k] for u in common], rng)
            paired[f"ours_minus_{other}_{k}"] = {"mean": mean, "ci95": [lo, hi], "boot_p_le0": p}
    out = {"config": {"n_perturb": N_PERT, "range": [LO, HI], "n_boot": N_BOOT, "leading_den_fixed": True,
                      "tau_perturbed": True, "spec_unchanged": True, "seed": "crc32(uid), shared across methods"},
           "summary": summary, "paired": paired, "per_task": res, "runtime_s": time.time() - t0}
    json.dump(out, open(f"{ROOT}/runs/robustness.json", "w"), indent=1)
    print(json.dumps({"summary": {m: {k: v for k, v in s.items() if k != "per_category"} for m, s in summary.items()},
                      "paired": paired, "runtime_s": out["runtime_s"]}, indent=1))

if __name__ == "__main__":
    main()
