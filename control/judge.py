"""Deterministic PID judge for ControlEval-style tasks.
Controller: K(s) = Kp + Ki/s + Kd*s/(Tf*s+1), unity feedback. Delay via 3rd-order Pade."""
import json, glob, math, warnings
import numpy as np, control as ct
warnings.filterwarnings("ignore")
TF_FILTER = 0.01  # derivative filter time constant (s)

def load_tasks():
    tasks = []
    for f in sorted(glob.glob(__file__.rsplit("/", 2)[0] + "/data/controlagent/ControlEval/*.json")):
        cat = f.split("/")[-1].replace("_data.json", "")
        for t in json.load(open(f)):
            t = dict(t); t["category"] = cat; t["uid"] = f"{cat}/{t['id']}"; tasks.append(t)
    return tasks

def plant(t):
    G = ct.tf(t["num"], t["den"])
    if t.get("tau"):
        n, d = ct.pade(t["tau"], 3); G = G * ct.tf(n, d)
    return G

def open_loop_unstable(t):
    return bool(np.any(np.real(np.roots(t["den"])) > 0))

def evaluate(t, Kp, Ki, Kd):
    """Returns dict of metrics + per-spec pass flags + success."""
    out = {"stable": False, "success": False}
    try:
        Kp, Ki, Kd = float(Kp), float(Ki), float(Kd)
        if not all(map(math.isfinite, (Kp, Ki, Kd))): return out
        s = ct.tf("s")
        K = Kp + Ki / s + Kd * s / (TF_FILTER * s + 1)
        L = plant(t) * K
        T = ct.feedback(L, 1)
        if np.any(np.real(ct.poles(T)) >= -1e-9): return out
        out["stable"] = True
        gm, pm, _, _ = ct.margin(L)
        gm_db = 20 * np.log10(gm) if np.isfinite(gm) and gm > 0 else float("inf")
        info = ct.step_info(T)  # auto time vector from system dynamics (same as ControlAgent)
        ts, os_ = info["SettlingTime"], info["Overshoot"]
        y_final = ct.dcgain(T); ess = abs(1 - y_final) if np.isfinite(y_final) else float("inf")
        out.update(gain_margin_db=gm_db, phase_margin=pm, settling_time=ts, overshoot=os_, ess=ess)
        checks = {
            "gm": True if open_loop_unstable(t) else gm_db >= t["gain_margin_min"],  # GM undefined-as-upper for unstable plants
            "pm": pm >= t["phase_margin_min"],
            "ts": t["settling_time_min"] <= ts <= t["settling_time_max"],
            "os": os_ <= t["overshoot_max"],
            "ess": ess <= max(t["steadystate_error_max"], 1e-4),
        }
        out["checks"] = checks; out["success"] = all(checks.values())
        out["score"] = sum(checks.values()) / len(checks)
    except Exception as e:
        out["error"] = str(e)[:120]
    return out

def task_prompt(t):
    delay = f" with input delay tau={t['tau']} s" if t.get("tau") else ""
    return (f"Plant G(s) = num/den with num={t['num']}, den={t['den']}{delay}.\n"
            f"Design PID gains for C(s)=Kp+Ki/s+Kd*s/({TF_FILTER}s+1) in unity feedback so that:\n"
            + ("" if open_loop_unstable(t) else f"- gain margin >= {t['gain_margin_min']:.2f} dB\n") + f"- phase margin >= {t['phase_margin_min']:.2f} deg\n"
            f"- 2% settling time between {t['settling_time_min']:.3f} and {t['settling_time_max']:.3f} s\n"
            f"- overshoot <= {t['overshoot_max']:.2f} %\n- steady-state error <= {max(t['steadystate_error_max'],1e-4)}\n"
            'Reply with JSON only: {"Kp": number, "Ki": number, "Kd": number}')
