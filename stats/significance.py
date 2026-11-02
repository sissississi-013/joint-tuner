"""Paired significance for PID results: exact McNemar + paired bootstrap 95% CI. Writes runs/significance.json"""
import json, numpy as np
from math import comb
def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if not x and y); b10 = sum(1 for x, y in zip(a, b) if x and not y); n = b01 + b10
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b01, b10) + 1)) / 2 ** n) if n else 1.0
    return b01, b10, p
def boot(a, b, R=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float); rng = np.random.default_rng(seed); n = len(a)
    idx = rng.integers(0, n, (R, n)); d = b[idx].mean(1) - a[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d <= 0).mean())
L = lambda f: json.load(open(f))
base = {r["uid"]: r["oks"] for r in L("runs/bon_base_n8.json")}
v3b = {r["uid"]: r["oks"] for r in L("runs/bon_q9b_v3_n8.json")}
v3g = {r["uid"]: bool(r["success"]) for r in L("runs/river_sft_q9b_v3.json")["final"]}
kimi = {r["uid"]: bool(r["success"]) for r in L("runs/pid_fireworks_kimi-k3_5_think.json")}
qwen = {r["uid"]: bool(r["success"]) for r in L("runs/pid_river_Qwen_Qwen3.8-27B-FP8_5.json")}
ens = {u: any(any(L_[u]) for L_ in [{r["uid"]: r["oks"] for r in L(f"runs/bon_q9b_{v}_n8.json")} for v in ("v1", "v2", "v3")]) for u in base}
U100 = sorted(base); U50 = sorted(kimi)
cmp = [
 ("Base 9B, 1 sample", {u: base[u][0] for u in U100}, "Tuned v3, greedy", v3g, U100),
 ("Base 9B, best-of-8 + sim", {u: any(base[u]) for u in U100}, "Tuned v3, best-of-8 + sim", {u: any(v3b[u]) for u in U100}, U100),
 ("Tuned v3, greedy", v3g, "Tuned v3, best-of-8 + sim", {u: any(v3b[u]) for u in U100}, U100),
 ("Kimi K3 thinking", kimi, "Tuned v3, greedy", v3g, U50),
 ("Kimi K3 thinking", kimi, "Tuned v3, best-of-8 + sim", {u: any(v3b[u]) for u in U50}, U50),
 ("Qwen3.8 27B", qwen, "Tuned v3, greedy", v3g, U50),
 ("Kimi K3 thinking", kimi, "3 ckpts x 8 + sim", ens, U50),
]
out = []
for na, A, nb, B, U in cmp:
    a = [bool(A[u]) for u in U]; b = [bool(B[u]) for u in U]
    b01, b10, p = mcnemar(a, b); lo, hi, pneg = boot(a, b)
    r = dict(before=na, after=nb, n=len(U), before_acc=sum(a) / len(U), after_acc=sum(b) / len(U), only_after=b01, only_before=b10, mcnemar_p=p, ci95=[lo, hi])
    out.append(r)
    print(f"{na:28s} {sum(a):3d}/{len(U)} -> {nb:28s} {sum(b):3d}/{len(U)} | only-after {b01:2d} only-before {b10:2d} | McNemar p={p:.2g} | diff 95% CI [{lo*100:+.0f}, {hi*100:+.0f}] pts")
json.dump(out, open("runs/significance.json", "w"), indent=1)
