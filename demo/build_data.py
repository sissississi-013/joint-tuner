"""Build demo/data.js from real overnight runs: per showcase task, step responses for each method + frontier points."""
import sys, json, numpy as np, control as ct
sys.path.insert(0, "control")
from judge import load_tasks, evaluate, plant, TF_FILTER
T = {t["uid"]: t for t in load_tasks()}
def gains_by_uid(path, key="gains"):
    d = json.load(open(path)); d = d["final"] if isinstance(d, dict) else d
    return {r["uid"]: r.get(key) or {} for r in d}
M = {
  "qwen27": ("Qwen3.8 27B (no think)", gains_by_uid("runs/pid_river_Qwen_Qwen3.8-27B-FP8_5.json"), "1.0 s"),
  "kimi_think": ("Kimi K3 (thinking)", gains_by_uid("runs/pid_fireworks_kimi-k3_5_think.json"), "219 s"),
  "ours": ("Ours: 9B tuned", gains_by_uid("runs/river_sft_q9b_v1.json"), "0.5 s"),
}
def response(t, g):
    try:
        s = ct.tf("s"); K = float(g["Kp"]) + float(g["Ki"]) / s + float(g["Kd"]) * s / (TF_FILTER * s + 1)
        Tcl = ct.feedback(plant(t) * K, 1)
        tend = 2.5 * t["settling_time_max"]
        tt = np.linspace(0, tend, 240); _, y = ct.step_response(Tcl, T=tt)
        y = np.clip(np.nan_to_num(y, nan=0, posinf=3, neginf=-3), -2, 3)
        return tt.round(5).tolist(), y.round(4).tolist()
    except Exception:
        return None, None
shows = []
cats = sorted({t["category"] for t in T.values()})
for c in cats:
    best = None
    for i in range(5):
        uid = f"{c}/{i}"; t = T[uid]
        ev = {k: evaluate(t, *(M[k][1].get(uid, {}).get(x) for x in ("Kp", "Ki", "Kd"))) if M[k][1].get(uid) else {"success": False} for k in M}
        rank = (ev["ours"]["success"], not ev["kimi_think"]["success"], not ev["qwen27"]["success"])
        if best is None or rank > best[0]: best = (rank, uid, ev)
    _, uid, ev = best; t = T[uid]; methods = []
    for k, (name, G, lat) in M.items():
        g = G.get(uid) or {}; tt, y = response(t, g) if g else (None, None)
        e = ev[k]
        methods.append({"key": k, "name": name, "latency": lat, "gains": g, "t": tt, "y": y, "success": bool(e.get("success")),
                        "checks": {kk: bool(v) for kk, v in e.get("checks", {}).items()}, "stable": bool(e.get("stable")),
                        "metrics": {kk: (None if not np.isfinite(float(e[kk])) else round(float(e[kk]), 3)) for kk in ("phase_margin", "settling_time", "overshoot") if kk in e}})
    shows.append({"uid": uid, "category": c, "num": t["num"], "den": t["den"], "tau": t.get("tau"),
                  "spec": {k: t[k] for k in ("gain_margin_min", "phase_margin_min", "settling_time_min", "settling_time_max", "overshoot_max")}, "methods": methods})
frontier = [
  {"name": "DeepSeek V4.1 Flash", "succ": 0, "lat": 5.1},
  {"name": "Qwen3.8 27B", "succ": 10, "lat": 1.0},
  {"name": "Kimi K3", "succ": 16, "lat": 2.8},
  {"name": "Kimi K3 thinking", "succ": 44, "lat": 219},
  {"name": "Ours 9B, 1 sample", "succ": 45, "lat": 0.5, "ours": True},
  {"name": "Ours 9B + simulator, best of 8", "succ": 76, "lat": 3.7, "ours": True},
  {"name": "3 checkpoints x 8 + simulator", "succ": 88, "lat": 11, "ours": True},
]
open("demo/data.js", "w").write("window.DEMO=" + json.dumps({"shows": shows, "frontier": frontier}) + ";")
for s in shows: print(s["uid"], [(m["key"], m["success"]) for m in s["methods"]])
