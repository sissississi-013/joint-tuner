"""Drone axis tasks: G(s) = k / (s^2 (tau s + 1))  (attitude or altitude with motor lag). Reverse-generated specs.
Usage: python control/gen_drone.py N OUT SEED"""
import sys, json, math, numpy as np
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from judge import evaluate, task_prompt
N, out, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]); rng = np.random.default_rng(seed); rows = []
LOOSE = dict(gain_margin_min=-1e9, phase_margin_min=-1e9, settling_time_min=0, settling_time_max=1e9, overshoot_max=1e9, steadystate_error_max=1e-4)
while len(rows) < N:
    axis = rng.choice(["roll", "pitch", "yaw", "altitude"])
    tau = rng.uniform(0.01, 0.08) if axis != "altitude" else rng.uniform(0.03, 0.15)
    k = rng.uniform(5, 200) if axis != "altitude" else rng.uniform(0.5, 15)
    t = {"num": [round(k / tau, 4)], "den": [1, round(1 / tau, 4), 0, 0], **LOOSE}
    found = None
    for _ in range(60):
        wc = rng.uniform(0.08, 0.35) / tau
        Kd = wc / (k) * rng.uniform(0.6, 1.6); Kp = Kd * wc / rng.uniform(1.5, 4); Ki = Kp * wc / rng.uniform(4, 15)
        r = evaluate(t, Kp, Ki, Kd)
        if r["stable"] and np.isfinite(r.get("settling_time", np.nan)) and r["phase_margin"] > 25 and r["overshoot"] < 35:
            found = (Kp, Ki, Kd, r); break
    if not found: continue
    Kp, Ki, Kd, r = found; ts = r["settling_time"]; gm = r["gain_margin_db"]
    t.update(gain_margin_min=round(min(gm - rng.uniform(0.5, 3), 10), 3) if np.isfinite(gm) else 6.0,
             phase_margin_min=round(r["phase_margin"] - rng.uniform(1, 10), 2), settling_time_min=round(ts * rng.uniform(0.01, 0.6), 5),
             settling_time_max=round(ts * rng.uniform(1.2, 2.5), 4), overshoot_max=round(max(r["overshoot"] + rng.uniform(1, 8), 5), 2),
             category=f"drone_{axis}", uid=f"drone/{len(rows)}")
    if not evaluate(t, Kp, Ki, Kd)["success"]: continue
    rows.append({"task": t, "label": {"Kp": float(f"{Kp:.4g}"), "Ki": float(f"{Ki:.4g}"), "Kd": float(f"{Kd:.4g}")}})
with open(out, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(len(rows), "drone tasks ->", out)
