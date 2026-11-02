"""Reverse generation: sample plant -> heuristic PID gains -> simulate -> specs = loosened achieved metrics.
Every task has a known-feasible label. Usage: python control/gen_synth.py N OUT.jsonl SEED"""
import sys, json, math, random, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from judge import evaluate, task_prompt, open_loop_unstable
import control as ct

def sample_plant(rng):
    kind = rng.choice(["fo_stable", "fo_unstable", "fo_delay", "so_stable", "so_unstable", "so_unstable_c", "higher"], p=[.18, .14, .12, .2, .1, .12, .14])
    if kind == "so_unstable_c":  # complex RHP pair: s^2 - 2*sig*s + (sig^2+w^2)
        sig = rng.uniform(0.2, 6); w = rng.uniform(0.3, 6)
        return kind, [rng.uniform(0.2, 20)], [1, -2 * sig, sig * sig + w * w], None
    k = rng.uniform(0.2, 20)
    if kind == "fo_stable":   return kind, [k], [1, rng.uniform(0.3, 20)], None
    if kind == "fo_unstable": return kind, [k], [1, -rng.uniform(0.3, 20)], None
    if kind == "fo_delay":    return kind, [k], [1, rng.uniform(0.4, 20)], round(rng.uniform(0.02, 1.2), 3)
    if kind == "so_stable":   return kind, [k], [1, rng.uniform(0.2, 9), rng.uniform(0.1, 25)], None
    if kind == "so_unstable": return kind, [k], [1, rng.uniform(-13, 20), rng.uniform(-300, -0.5)], None
    n = rng.choice([3, 4, 5]); poles = -np.exp(rng.uniform(np.log(0.3), np.log(60), n))
    den = np.real(np.poly(poles)); gain = float(np.prod(-poles)) * rng.uniform(0.3, 3)
    return kind, [round(gain, 3)], [round(float(x), 4) for x in den], None

def heuristic_gains(rng, t):
    G = ct.tf(t["num"], t["den"])
    if t.get("tau"): n, d = ct.pade(t["tau"], 3); G = G * ct.tf(n, d)
    poles = np.abs(np.roots(t["den"])); base = float(np.median(poles)) if len(poles) else 1.0
    if t.get("tau"): base = min(base, 0.6 / t["tau"])
    unstable = bool(np.any(np.real(np.roots(t["den"])) > 0))
    if unstable:
        base = float(np.max(poles)); wc = base * math.exp(rng.uniform(math.log(2), math.log(30)))
    else:
        wc = base * math.exp(rng.uniform(math.log(0.3), math.log(30 if not t.get("tau") else 4)))
    mag = abs(complex(ct.evalfr(G, 1j * wc)))
    Kp = (1 / mag) * rng.uniform(0.5, 1.5)
    Ki = Kp * wc / (rng.uniform(5, 30) if unstable else rng.uniform(1.5, 10))
    if unstable and len(t["den"]) == 3: Kd = Kp / wc * rng.uniform(0.3, 1.5)
    else: Kd = Kp / wc * (rng.uniform(0.05, 0.8) if rng.random() < 0.5 else 0.0)
    return Kp, Ki, Kd

def make(rng, i):
    for _ in range(40):
        kind, num, den, tau = sample_plant(rng)
        t = {"num": num, "den": den, "steadystate_error_max": 0.01 if tau else 0.0001}
        if tau: t["tau"] = tau
        t.update(gain_margin_min=-1e9, phase_margin_min=-1e9, settling_time_min=0, settling_time_max=1e9, overshoot_max=1e9)
        good = None
        for _try in range(40 if kind.startswith("so_unstable") else 6):
            Kp, Ki, Kd = heuristic_gains(rng, t)
            r = evaluate(t, Kp, Ki, Kd)
            if not r["stable"] or not np.isfinite(r.get("settling_time", np.nan)): continue
            if r["phase_margin"] < 20 or r["overshoot"] > 30: continue
            good = r; break
        if good is None: continue
        r = good
        if not open_loop_unstable(t) and not (np.isfinite(r["gain_margin_db"]) or r["gain_margin_db"] == float("inf")): continue
        ts = r["settling_time"]
        gm = r["gain_margin_db"]
        t.update(
            gain_margin_min=round(min(gm - rng.uniform(0.5, 4), 10), 3) if np.isfinite(gm) else round(rng.uniform(5, 10), 3),
            phase_margin_min=round(r["phase_margin"] - rng.uniform(1, 12), 3),
            settling_time_min=round(ts * math.exp(rng.uniform(math.log(0.001), math.log(0.7))), 6),
            settling_time_max=round(ts * rng.uniform(1.15, 3.0), 5),
            overshoot_max=round(max(r["overshoot"] + rng.uniform(1, 8), 5.0), 3),
            category=kind, uid=f"synth/{i}")
        chk = evaluate(t, Kp, Ki, Kd)
        if not chk["success"]: continue
        return t, {"Kp": float(f"{Kp:.4g}"), "Ki": float(f"{Ki:.4g}"), "Kd": float(f"{Kd:.4g}")}
    return None

if __name__ == "__main__":
    N, out, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    rng = np.random.default_rng(seed); rows = []; from collections import Counter; c = Counter()
    while len(rows) < N:
        m = make(rng, len(rows))
        if m is None: continue
        t, g = m; c[t["category"]] += 1
        rows.append({"task": t, "label": g,
                     "messages": [{"role": "system", "content": "You are an expert control engineer tuning PID loops for robot joint actuators."},
                                  {"role": "user", "content": task_prompt(t)},
                                  {"role": "assistant", "content": json.dumps(g)}]})
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(len(rows), "rows ->", out, dict(c))
