"""Make the README figures from raw per-task results. Run from the buildathon workspace or the repo root (needs runs/).
Outputs docs/figures/*.png"""
import json, math, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from math import comb
OUT = os.environ.get("FIGDIR", "docs/figures"); os.makedirs(OUT, exist_ok=True)
for f in ("~/Library/Fonts/Aileron-Regular.ttf", "~/Library/Fonts/Aileron-Bold.ttf"):
    p = os.path.expanduser(f)
    if os.path.exists(p): fm.fontManager.addfont(p)
BG, INK, MU, GRID = "#160c24", "#f6f0fc", "#aa9bc3", "#34264a"
PUR, PINK, OR, GREY, OAI, FWC = "#8b5cf6", "#ff4fa3", "#ff8a4c", "#968caa", "#10a37f", "#6720ff"
plt.rcParams.update({"font.family": "Aileron" if any("Aileron" in f.name for f in fm.fontManager.ttflist) else "sans-serif",
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": MU, "xtick.color": MU, "ytick.color": MU,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False})
R = "runs"
tr = json.load(open(f"{R}/trace_v3.json")); rows = {r["uid"]: r for r in tr["rows"]}; U = sorted(rows)
v2 = {r["uid"]: r for r in json.load(open(f"{R}/trace_v3_repair2.json"))["rows"]}
gpt = {r["uid"]: bool(r["success"]) for r in json.load(open(f"{R}/pid_closed_gpt-6.1-terra_10.json"))}
rep = {r["uid"]: bool(r["sample_ok"] or r["repair_ok"]) for r in json.load(open(f"{R}/repair_v3_n8_stab.json"))}
cs = {r["uid"]: bool(r["success"]) for r in json.load(open(f"{R}/coldstart_150x3.json"))}
fa = {r["uid"]: r for r in json.load(open(f"{R}/fw_sft_q36v3.json"))["after"]}
fb = {r["uid"]: r for r in json.load(open(f"{R}/fw_sft_q36v3_before_eval.json"))["before"]}
kimi_rows = json.load(open(f"{R}/pid_fireworks_kimi-k3_think_full100.json"))
kimi = {r["uid"]: bool(r.get("success")) for r in (kimi_rows if isinstance(kimi_rows, list) else kimi_rows.get("rows", []))}
at = lambda r, k, B: bool(r[k] and r[k] <= B)
def vec(d): return np.array([bool(d.get(u, False)) for u in U])
def mcn(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum()); n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n); return x, y, p
rng = np.random.default_rng(0)
def boot(a, b, n=10000):
    idx = rng.integers(0, len(a), (n, len(a))); return (a[idx].mean(1) - b[idx].mean(1)) * 100
C = [  # label, A, B, color
 ("Fine-tuned vs base, first answer (Fireworks 35B)", vec({u: fa[u]["greedy"]["oks"][0] for u in fa}), vec({u: fb[u]["greedy"]["oks"][0] for u in fb}), FWC),
 ("Fine-tuned vs base, best of 8 (Fireworks 35B)", vec({u: any(fa[u]["bon"]["oks"]) for u in fa}), vec({u: any(fb[u]["bon"]["oks"]) for u in fb}), FWC),
 ("Joint Tuner vs optimizer only, 1 test", vec({u: at(rows[u], "ours_first", 1) for u in U}), vec({u: at(rows[u], "cold_first", 1) for u in U}), OR),
 ("Joint Tuner vs optimizer only, 8 tests", vec({u: at(rows[u], "ours_first", 8) for u in U}), vec({u: at(rows[u], "cold_first", 8) for u in U}), OR),
 ("Joint Tuner vs optimizer only, 256 tests", vec({u: at(rows[u], "ours_first", 256) for u in U}), vec({u: at(rows[u], "cold_first", 256) for u in U}), OR),
 ("Joint Tuner (repair v2) vs optimizer only, 600 tests", vec({u: at(v2[u], "ours_first", 600) for u in U}), vec({u: at(rows[u], "cold_first", 600) for u in U}), GREY),
 ("Joint Tuner repair vs optimizer, matched budget", vec(rep), vec(cs), OR),
 ("Joint Tuner repair vs GPT-6.1 Terra", vec(rep), vec(gpt), OAI),
 ("Joint Tuner repair vs Kimi K3 thinking", vec(rep), vec(kimi), GREY),
]
res = []
for lab, a, b, c in C:
    x, y, p = mcn(a, b); bs = boot(a, b); res.append(dict(label=lab, a=int(a.sum()), b=int(b.sum()), a_only=x, b_only=y, p=p,
        diff=float((a.mean() - b.mean()) * 100), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), color=c, boots=bs))
json.dump([{k: v for k, v in r.items() if k != "boots"} for r in res], open(f"{OUT}/comparisons.json", "w"), indent=1)
for r in res: print(f"{r['label']:55s} {r['a']:3d} vs {r['b']:3d} | {r['a_only']:2d}/{r['b_only']:2d} p={r['p']:.2g} | {r['diff']:+.0f} [{r['lo']:+.0f},{r['hi']:+.0f}]")
def pfmt(p): return "p < 1e-10" if p < 1e-10 else (f"p = {p:.0e}" if p < 1e-3 else f"p = {p:.2g}")
# 1 McNemar disagreements
fig, ax = plt.subplots(figsize=(12, 6.2)); ax.grid(axis="y")
y = np.arange(len(res))[::-1]
for yi, r in zip(y, res):
    ax.barh(yi + .18, r["a_only"], .34, color=r["color"]); ax.barh(yi - .18, r["b_only"], .34, color=GREY, alpha=.55)
    ax.text(max(r["a_only"], r["b_only"]) + 1, yi, f"{r['a_only']} vs {r['b_only']}   {pfmt(r['p'])}", va="center", color=PINK if r["p"] < .05 else MU, fontsize=11, fontweight="bold")
ax.set_yticks(y, [r["label"] for r in res]); ax.set_xlabel("tasks where exactly one system was right (colored = first system, grey = second)")
ax.set_title("McNemar: who wins the disagreements?", loc="left", color=INK, fontsize=18, fontweight="bold", pad=14); ax.set_xlim(0, 75)
fig.tight_layout(); fig.savefig(f"{OUT}/mcnemar.png", dpi=150); plt.close(fig)
# 2 forest plot
fig, ax = plt.subplots(figsize=(12, 6.2)); ax.grid(axis="x")
for yi, r in zip(y, res):
    ax.plot([r["lo"], r["hi"]], [yi, yi], color=r["color"], lw=4, solid_capstyle="round"); ax.plot(r["diff"], yi, "o", color=INK, ms=8)
    ax.text(62, yi, f"{r['diff']:+.0f} pts  [{r['lo']:+.0f}, {r['hi']:+.0f}]", va="center", color=PINK if r["lo"] > 0 or r["hi"] < 0 else MU, fontsize=11, fontweight="bold")
ax.axvline(0, color=MU, lw=1.5, ls="--"); ax.set_yticks(y, [r["label"] for r in res]); ax.set_xlim(-20, 90)
ax.set_xlabel("difference in tasks solved, points (paired bootstrap 95% CI, 10,000 resamples of the 100 tasks)")
ax.set_title("Bootstrap: how big is each gain, and does it exclude zero?", loc="left", color=INK, fontsize=18, fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(f"{OUT}/forest.png", dpi=150); plt.close(fig)
# 3 bootstrap distributions for two headline comparisons
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax, r in zip(axs, [res[3], res[7]]):
    ax.hist(r["boots"], bins=np.arange(-20, 75, 1), color=r["color"], alpha=.9); ax.axvline(0, color=MU, ls="--")
    for q in (r["lo"], r["hi"]): ax.axvline(q, color=PINK, lw=2)
    share = (r["boots"] > 0).mean() * 100
    ax.set_title(r["label"], loc="left", color=INK, fontsize=13, fontweight="bold")
    ax.text(.02, .92, f"95% CI [{r['lo']:+.0f}, {r['hi']:+.0f}] pts\nahead in {share:.0f}% of 10,000 resampled test sets", transform=ax.transAxes, color=PINK, fontsize=11, va="top", fontweight="bold")
    ax.set_xlabel("difference in tasks solved, points"); ax.set_yticks([])
fig.suptitle("Would the gap survive a different test set?", x=.01, ha="left", color=INK, fontsize=18, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT}/bootstrap.png", dpi=150); plt.close(fig)
# 4 budget curve
B = tr["budgets"]; cur = lambda d, k: [sum(at(d[u], k, b) for u in U) for b in B]
o, c, o2 = cur(rows, "ours_first"), cur(rows, "cold_first"), cur(v2, "ours_first")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(B, c, "-o", color=GREY, lw=3, label="Optimizer only (Nelder-Mead, random restarts)")
ax.plot(B, o2, "-o", color=OR, lw=4, label="Joint Tuner (9B proposes 8, simulator checks, repair v2)")
ax.plot(1, sum(gpt.values()), "o", color=OAI, ms=13, label="GPT-6.1 Terra, one answer (18.7 s thinking)")
ax.plot(1.06, sum(kimi.values()), "s", color=PUR, ms=11, label="Kimi K3 thinking, one answer (222 s)")
ax.set_xscale("log", base=2); ax.set_xticks(B, [str(b) for b in B]); ax.set_ylim(0, 104)
ax.set_xlabel("physical step tests allowed per task (log scale)"); ax.set_ylabel("tasks solved out of 100")
ax.annotate("8 tests: 76 vs 27\n+49 pts, p = 3e-13", xy=(8, 76), xytext=(2.3, 93), color=PINK, fontsize=12, fontweight="bold", arrowprops=dict(arrowstyle="-", color=PINK))
ax.vlines(8, c[3], o[3], color=PINK, lw=2.5)
ax.annotate("1 test: 46 vs 19", xy=(1, 46), xytext=(1.08, 34), color=PINK, fontsize=12, fontweight="bold")
ax.text(1.05, 5, "median tests to first success: 1 (ours) vs 19.5 (optimizer)", color=INK, fontsize=12)
ax.annotate("600 tests: 93 vs 93, same 93 tasks", xy=(600, 93), xytext=(70, 98), color=MU, fontsize=11)
ax.annotate("GPT-6.1 Terra 89", xy=(1, 89), xytext=(1.08, 84), color=OAI, fontsize=11, fontweight="bold")
ax.annotate("Kimi K3 thinking 90", xy=(1, 90), xytext=(1.08, 77), color=PUR, fontsize=11, fontweight="bold")
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.1), facecolor=BG, edgecolor=GRID, labelcolor=INK, fontsize=11)
ax.set_title("Tasks solved vs. physical tests allowed (100 held-out ControlEval tasks)", loc="left", color=INK, fontsize=17, fontweight="bold", pad=14)
fig.tight_layout(); fig.savefig(f"{OUT}/budget_curve.png", dpi=150); plt.close(fig)
print("figures ->", OUT)
