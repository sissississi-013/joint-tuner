"""Build demo/race_data.js from runs/trace_v3.json (+ GPT-6.1 Terra answers): per showcase task, every trial's step response for each tuner."""
import sys, json, numpy as np, control as ct
sys.path.insert(0, "control")
from judge import load_tasks, evaluate, plant, TF_FILTER
T = {t["uid"]: t for t in load_tasks()}
tr = json.load(open("runs/trace_v3.json")); gpt = {r["uid"]: r for r in json.load(open("runs/pid_closed_gpt-6.1-terra_10.json"))}
s = ct.tf("s")
def resp(t, g, n=110):
    try:
        Tcl = ct.feedback(plant(t) * (g[0] + g[1] / s + g[2] * s / (TF_FILTER * s + 1)), 1)
        tt = np.linspace(0, 2.5 * t["settling_time_max"], n); _, y = ct.step_response(Tcl, T=tt)
        return np.clip(np.nan_to_num(y, nan=3, posinf=3, neginf=-2), -2, 3).round(3).tolist()
    except Exception: return [3.0] * n
NAMES = {"first_order_stable_fast": "Fast servo", "first_order_stable_moderate": "Conveyor drive", "first_order_stable_slow": "Heavy turntable",
         "first_order_unstable": "Balancing platform", "first_order_w_delay": "Motor with sensor lag", "higher_order": "Flexible arm",
         "second_order_stable_fast": "Wrist joint", "second_order_stable_moderate": "Elbow joint", "second_order_stable_slow": "Shoulder joint", "second_order_unstable": "Inverted pendulum"}
shows = []
for cat in sorted(NAMES):
    rows = [r for r in tr["rows"] if r["category"] == cat]
    # showcase: ours solves; prefer the largest gap in trials vs cold, then GPT fails
    def key(r):
        o = r["ours_first"] or 10**6; c = r["cold_first"] or 700
        return (bool(r["ours_first"]), c - o, not gpt[r["uid"]]["success"])
    r = max(rows, key=key); t = T[r["uid"]]
    def lane(log, first, cap=160):
        k = first or len(log); idx = list(range(min(k, len(log))))
        if len(idx) > cap: idx = sorted(set(np.linspace(0, len(idx) - 1, cap).astype(int).tolist()))
        return {"trials": first or len(log), "solved": bool(first), "shown": [{"i": i + 1, "g": log[i][:3], "ok": log[i][3], "y": resp(t, log[i][:3])} for i in idx]}
    g = gpt[r["uid"]]; gg = g.get("gains") or {}
    try: gv = [float(gg["Kp"]), float(gg["Ki"]), float(gg["Kd"])]
    except Exception: gv = None
    gl = {"trials": 1, "solved": bool(g["success"]), "think_s": round(g.get("latency_s", 0), 1),
          "shown": [{"i": 1, "g": gv, "ok": bool(g["success"]), "y": resp(t, gv) if gv else [0.0] * 110}]}
    p = t
    shows.append({"uid": r["uid"], "name": NAMES[cat], "category": cat, "num": t["num"], "den": t["den"], "tmax": 2.5 * t["settling_time_max"],
                  "spec": {k: p[k] for k in ("phase_margin_min", "settling_time_min", "settling_time_max", "overshoot_max")},
                  "lanes": {"cold": lane(r["cold_log"], r["cold_first"]), "gpt": gl, "ours": lane(r["ours_log"], r["ours_first"])}})
    print(cat, r["uid"], "ours", r["ours_first"], "cold", r["cold_first"], "gpt", g["success"])
out = {"shows": shows, "budgets": tr["budgets"], "curve": tr["curve"], "gpt_total": sum(x["success"] for x in gpt.values())}
open("demo/race_data.js", "w").write("window.RACE=" + json.dumps(out, separators=(",", ":")) + ";")
print("bytes", len(json.dumps(out)))
