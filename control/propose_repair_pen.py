import numpy as np
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
