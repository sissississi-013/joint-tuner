"""README figures, drawn from the raw per-task results in runs/.
Usage (repo root): python stats/figures.py   -> docs/figures/*.png"""
import json, os, statistics as st
from math import comb
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.environ.get("FIGDIR", "docs/figures"); os.makedirs(OUT, exist_ok=True)
for f in ("/System/Library/Fonts/Supplemental/Charter.ttc", "/System/Library/Fonts/HelveticaNeue.ttc"):
    if os.path.exists(f):
        try: fm.fontManager.addfont(f)
        except Exception: pass
names = {f.name for f in fm.fontManager.ttflist}
SERIF = "Charter" if "Charter" in names else "DejaVu Serif"
SANS = "Helvetica Neue" if "Helvetica Neue" in names else "DejaVu Sans"

# palette: ivory paper, slate ink, clay accent, muted secondaries
IVORY, OAT, CLOUD = "#FAF9F5", "#F0EEE6", "#E8E6DC"
SLATE, INK2, INK3 = "#141413", "#5E5D59", "#87867F"
CLAY, CLAYD = "#D97757", "#B5573A"
OLIVE, SKY, KRAFT, BASE = "#788C5D", "#6A9BCC", "#D4A27F", "#B0AEA5"
plt.rcParams.update({
    "font.family": SANS, "font.size": 10, "text.color": SLATE,
    "figure.facecolor": IVORY, "axes.facecolor": IVORY, "savefig.facecolor": IVORY,
    "axes.edgecolor": INK3, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.labelsize": 9.5,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "grid.color": CLOUD, "grid.linewidth": 0.8, "axes.unicode_minus": False, "legend.frameon": False})

def frame(title, subtitle, source, size=(9, 5.2), rect=(0.08, 0.15, 0.88, 0.62)):
    fig = plt.figure(figsize=size)
    fig.text(0.04, 0.94, title, family=SERIF, fontsize=16, color=SLATE, va="top")
    fig.text(0.04, 0.865, subtitle, fontsize=10, color=INK2, va="top")
    fig.text(0.04, 0.035, source, fontsize=7.8, color=INK3, va="bottom")
    ax = fig.add_axes(rect) if rect else None
    return fig, ax

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=200); plt.close(fig); print("wrote", name)

# ---------- data ----------
R = "runs"
J = lambda f: json.load(open(f"{R}/{f}"))
tr = J("trace_v3.json"); rows = {r["uid"]: r for r in tr["rows"]}; U = sorted(rows)
v2 = {r["uid"]: r for r in J("trace_v3_repair2.json")["rows"]}
gpt = {r["uid"]: bool(r["success"]) for r in J("pid_closed_gpt-6.1-terra_10.json")}
rep = {r["uid"]: bool(r["sample_ok"] or r["repair_ok"]) for r in J("repair_v3_n8_stab.json")}
cs = {r["uid"]: bool(r["success"]) for r in J("coldstart_150x3.json")}
fa = {r["uid"]: r for r in J("fw_sft_q36v3.json")["after"]}
fb = {r["uid"]: r for r in J("fw_sft_q36v3_before_eval.json")["before"]}
kr = J("pid_fireworks_kimi-k3_think_full100.json"); kr = kr if isinstance(kr, list) else kr.get("rows", [])
kimi = {r["uid"]: bool(r.get("success")) for r in kr}
bon_base = J("bon_base_n8.json"); bon8 = J("bon_q9b_v3_n8.json"); bon32 = J("bon_q9b_v3_n32.json")
sft9 = J("river_sft_q9b_v3.json")["log"]["evals"][-1]["success"]
rob = J("robustness.json")["paired"]

at = lambda r, k, B: bool(r[k] and r[k] <= B)
vec = lambda d: np.array([bool(d.get(u, False)) for u in U])
def mcn(a, b):
    x, y = int((a & ~b).sum()), int((~a & b).sum()); n = x + y
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n)
    return x, y, p
rng = np.random.default_rng(0)
def boot(a, b, n=10000):
    idx = rng.integers(0, len(a), (n, len(a))); return (a[idx].mean(1) - b[idx].mean(1)) * 100
def pfmt(p):
    if p >= 0.01: return f"p = {p:.2f}".rstrip("0").rstrip(".") if p < 1 else "p = 1.0"
    m, e = f"{p:.0e}".split("e"); return f"p = {m}e-{int(e[1:])}"

C = [("Fireworks 35B, tuned vs base, first answer", vec({u: fa[u]["greedy"]["oks"][0] for u in fa}), vec({u: fb[u]["greedy"]["oks"][0] for u in fb})),
     ("Fireworks 35B, tuned vs base, best of 8", vec({u: any(fa[u]["bon"]["oks"]) for u in fa}), vec({u: any(fb[u]["bon"]["oks"]) for u in fb})),
     ("Joint Tuner vs optimizer, 1 step test", vec({u: at(rows[u], "ours_first", 1) for u in U}), vec({u: at(rows[u], "cold_first", 1) for u in U})),
     ("Joint Tuner vs optimizer, 8 step tests", vec({u: at(rows[u], "ours_first", 8) for u in U}), vec({u: at(rows[u], "cold_first", 8) for u in U})),
     ("Joint Tuner vs optimizer, 256 step tests", vec({u: at(rows[u], "ours_first", 256) for u in U}), vec({u: at(rows[u], "cold_first", 256) for u in U})),
     ("Joint Tuner vs optimizer, matched compute", vec(rep), vec(cs)),
     ("Joint Tuner (v2) vs optimizer, 600 step tests", vec({u: at(v2[u], "ours_first", 600) for u in U}), vec({u: at(rows[u], "cold_first", 600) for u in U})),
     ("Joint Tuner vs GPT-6.1 Terra", vec(rep), vec(gpt)),
     ("Joint Tuner vs Kimi K3 thinking", vec(rep), vec(kimi))]
res = []
for lab, a, b in C:
    x, y, p = mcn(a, b); bs = boot(a, b)
    res.append(dict(label=lab, a=int(a.sum()), b=int(b.sum()), won=x, lost=y, p=p, diff=float((a.mean() - b.mean()) * 100),
                    lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), boots=bs))
json.dump([{k: v for k, v in r.items() if k != "boots"} for r in res], open(f"{OUT}/comparisons.json", "w"), indent=1)

# ---------- 1. pipeline diagram ----------
def box(ax, x, y, w, h, title, body, fc=OAT, ec=INK3, tc=SLATE, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.018", fc=fc, ec=ec, lw=lw))
    ax.text(x + 0.018, y + h - 0.035, title, family=SERIF, fontsize=12, color=tc, va="top")
    ax.text(x + 0.018, y + h - 0.105, body, fontsize=8.6, color=INK2 if tc == SLATE else tc, va="top", linespacing=1.45)
def arrow(ax, p, q, text=None, tx=None, color=INK2, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.1, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2))
    if text: ax.text(*(tx or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.03)), text, fontsize=8.2, color=color, ha="center")

fig, _ = frame("Propose, simulate, repair", "The model makes educated guesses, the simulator throws out bad ones before any hardware moves, and an optimizer polishes the rest.",
               "Model: Qwen3.6-35B-A3B with a rank-16 LoRA on Fireworks (pipeline runs also used a Qwen3.5-9B LoRA). Judge: control/judge.py. Repair: control/repair_v2.py.", size=(10, 5.6), rect=None)
ax = fig.add_axes((0.03, 0.09, 0.94, 0.74)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
box(ax, 0.00, 0.60, 0.19, 0.30, "The task", "Plant model of one joint\nSpec: phase and gain margin,\nsettling time, overshoot,\nsteady-state error")
box(ax, 0.25, 0.60, 0.22, 0.30, "1  Propose", "Fine-tuned open model\nwrites 8 gain sets in\none call (1.5 s median\non Fireworks)", fc=CLAY, ec=CLAY, tc=IVORY)
box(ax, 0.53, 0.60, 0.21, 0.30, "2  Simulate", "Control-theory judge\nchecks every set against\nthe spec. No hardware\nmoves yet.")
box(ax, 0.80, 0.60, 0.20, 0.30, "Run on hardware", "Passing gains go to the\nreal joint. Median: 1 step\ntest to first success,\nvs 19.5 for the optimizer")
box(ax, 0.53, 0.10, 0.21, 0.30, "3  Repair", "No set passes? Nelder-Mead\nstarts from the best ones\nand searches in log-gain\nspace, within the budget")
box(ax, 0.00, 0.10, 0.40, 0.30, "Training (once, on Fireworks)", "7,500 designs made in reverse: sample gains,\nsimulate them, write down the spec they meet.\nEvery label is feasible by construction.\nLoRA fine-tune, 300 steps, $5.11.", fc=IVORY)
arrow(ax, (0.19, 0.75), (0.25, 0.75))
arrow(ax, (0.47, 0.75), (0.53, 0.75))
arrow(ax, (0.74, 0.75), (0.80, 0.75), "passes", color=OLIVE)
arrow(ax, (0.62, 0.60), (0.62, 0.40), "none pass", tx=(0.575, 0.50), color=CLAYD)
arrow(ax, (0.68, 0.40), (0.68, 0.60), "re-check", tx=(0.725, 0.50))
arrow(ax, (0.30, 0.40), (0.33, 0.60), "trains the model", tx=(0.37, 0.48))
save(fig, "pipeline")

# ---------- 2. leaderboard ----------
L = [("Joint Tuner, repair v2, up to 600 step tests", 93, CLAY), ("Optimizer only, up to 600 step tests", 93, INK3),
     ("Joint Tuner, 8 samples + repair", 90, CLAY), ("Kimi K3, thinking (222 s per task)", 90, OLIVE),
     ("GPT-6.1 Terra, one answer (18.7 s)", 89, SKY), ("Joint Tuner, best of 32", 89, CLAY),
     ("Optimizer only, matched compute", 84, INK3), ("Joint Tuner on Fireworks 35B, best of 8", 81, CLAY),
     ("Kimi K3, no thinking", 22, BASE), ("Qwen3.8 27B", 11, BASE), ("Base Qwen3.5 9B, one answer", 8, BASE), ("Nemotron 3.5 Super", 1, BASE)]
fig, ax = frame("Joint Tuner matches the frontier on 100 held-out tasks",
                "Tasks solved out of 100 ControlEval tasks, same judge and prompt for every system. Clay bars are ours.",
                "Ours vs GPT-6.1 Terra: 90 vs 89, McNemar p = 1.0. Ours vs Kimi K3 thinking: 90 vs 90, p = 1.0. A tie, not a win. Sources: runs/*.json.",
                size=(9, 6), rect=(0.36, 0.11, 0.58, 0.70))
y = np.arange(len(L))[::-1]
ax.barh(y, [v for _, v, _ in L], color=[c for *_, c in L], height=0.64)
for yi, (lab, v, c) in zip(y, L):
    ax.text(v + 1, yi, str(v), va="center", fontsize=9.5, color=SLATE, weight="bold" if c == CLAY else "normal")
ax.set_yticks(y, [l for l, _, _ in L]); ax.set_xlim(0, 100); ax.set_xticks([0, 25, 50, 75, 100])
for t, (_, _, c) in zip(ax.get_yticklabels(), L): t.set_color(SLATE if c == CLAY else INK2)
ax.spines["left"].set_visible(False); ax.xaxis.grid(True); ax.set_axisbelow(True)
save(fig, "leaderboard")

# ---------- 3. budget curve ----------
B = tr["budgets"]; cur = lambda d, k: [sum(at(d[u], k, b) for u in U) for b in B]
o, c, o2 = cur(rows, "ours_first"), cur(rows, "cold_first"), cur(v2, "ours_first")
fig, ax = frame("With 8 physical tests, Joint Tuner solves 76 tasks. The optimizer solves 27.",
                "Tasks solved out of 100 when each task may run at most N step tests on the joint.",
                "Joint Tuner: Qwen3.5-9B LoRA proposes 8, simulator checks, repair v2. Optimizer: Nelder-Mead with random restarts. Gaps from 1 to 256 tests are significant\nafter Holm correction. Sources: runs/trace_v3.json, runs/trace_v3_repair2.json.",
                size=(9, 5.4), rect=(0.08, 0.17, 0.74, 0.62))
ax.plot(B, c, color=INK3, lw=2, marker="o", ms=4); ax.plot(B, o2, color=CLAY, lw=2.6, marker="o", ms=4.5)
ax.set_xscale("log", base=2); ax.set_xticks(B, [str(b) for b in B]); ax.set_ylim(0, 100); ax.set_xlim(0.8, 700)
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_xlabel("step tests allowed per task (log scale)"); ax.set_ylabel("tasks solved")
ax.text(720, o2[-1] + 1.5, "Joint Tuner  93", color=CLAY, fontsize=9.5, weight="bold", va="bottom", clip_on=False)
ax.text(720, c[-1] - 3, "Optimizer only  93", color=INK2, fontsize=9.5, va="top", clip_on=False)
ax.vlines(8, c[3], o2[3], color=SLATE, lw=1); ax.plot([8, 8], [c[3], o2[3]], "_", color=SLATE, ms=8)
ax.text(7.2, 36, f"+{o2[3]-c[3]} points at 8 tests\n95% CI +38 to +59\nMcNemar p = 3e-13", fontsize=8.8, va="center", ha="right")
ax.text(1.1, 11, f"At 1 test: {o2[0]} vs {c[0]}", fontsize=8.8, color=SLATE)
ax.plot([1], [sum(gpt.values())], "D", color=SKY, ms=6); ax.text(1.15, sum(gpt.values()) - 0.5, "GPT-6.1 Terra, one answer: 89", color=SKY, fontsize=8.6, va="center")
ax.plot([1], [sum(kimi.values()) + 2.5], "s", color=OLIVE, ms=6); ax.text(1.15, sum(kimi.values()) + 3.2, "Kimi K3 thinking, one answer: 90", color=OLIVE, fontsize=8.6, va="center")
ax.text(40, 12, "Median step tests to first success:\n1 for Joint Tuner, 19.5 for the optimizer", fontsize=8.8, color=SLATE)
save(fig, "budget_curve")

# ---------- 4. speed ----------
med = lambda xs: st.median(xs)
P = [("Joint Tuner 9B, 8 samples + repair", 6.3, 90, CLAY, "above"), ("Joint Tuner 9B, best of 32", 4.9, 89, CLAY, "left"),
     ("Joint Tuner 35B on Fireworks, best of 8", med(r["bon"]["latency_s"] for r in fa.values()), sum(any(r["bon"]["oks"]) for r in fa.values()), CLAY, "right"),
     ("Base 35B on Fireworks, best of 8", med(r["bon"]["latency_s"] for r in fb.values()), sum(any(r["bon"]["oks"]) for r in fb.values()), BASE, "right"),
     ("GPT-6.1 Terra", 18.7, 89, SKY, "right"), ("Kimi K3 thinking", 222, 90, OLIVE, "above")]
fig, ax = frame("Same accuracy as GPT-6.1 Terra at a third of the model time",
                "Tasks solved out of 100 vs median seconds of model time per task, before any physical test.",
                "Fireworks latencies are measured per call in runs/fw_sft_q36v3*.json. GPT-6.1 Terra and Kimi K3 latencies are medians from their result files.",
                size=(9, 5.2), rect=(0.08, 0.15, 0.86, 0.64))
for lab, x, yv, col, where in P:
    ax.plot(x, yv, "o", color=col, ms=8, mec=IVORY, mew=1)
    kw = dict(fontsize=8.8, color=SLATE if col == CLAY else INK2, va="center")
    if where == "above": ax.text(x, yv + 2.6, f"{lab}: {yv}", ha="center", **kw)
    elif where == "left": ax.text(x / 1.12, yv - 0.6, f"{lab}: {yv}", ha="right", **kw)
    else: ax.text(x * 1.12, yv, f"{lab}: {yv}", ha="left", **kw)
ax.annotate("", (6.3, 85.5), (18.7, 85.5), arrowprops=dict(arrowstyle="<-", color=INK3, lw=0.8))
ax.text(10.8, 83.6, "3 times less model time", fontsize=8.2, color=INK2, ha="center", va="center")
ax.set_xscale("log"); ax.set_xlim(0.8, 600); ax.set_ylim(45, 100); ax.set_xticks([1, 3, 10, 30, 100, 300], ["1 s", "3 s", "10 s", "30 s", "100 s", "300 s"])
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_xlabel("median model time per task (log scale)"); ax.set_ylabel("tasks solved")
save(fig, "speed")

# ---------- 5. fireworks before / after ----------
fg = res[0]; fbn = res[1]
fig, ax = frame("Fine-tuning on Fireworks took the 35B model from 20 to 50",
                "Qwen3.6-35B-A3B on the same 100 held-out tasks, before and after a $5.11 LoRA fine-tune on Fireworks serverless training.",
                "LoRA rank 16, learning rate 1e-4, 300 steps of 32 examples. Sources: runs/fw_sft_q36v3_before_eval.json, runs/fw_sft_q36v3.json.",
                size=(9, 5.2), rect=(0.08, 0.15, 0.86, 0.62))
xs = np.array([0, 1.25]); w = 0.36
ax.bar(xs - w / 2 - 0.02, [fg["b"], fbn["b"]], w, color=BASE); ax.bar(xs + w / 2 + 0.02, [fg["a"], fbn["a"]], w, color=CLAY)
for x0, r in zip(xs, (fg, fbn)):
    ax.text(x0 - w / 2 - 0.02, r["b"] + 1.5, f"{r['b']}\nbase", ha="center", fontsize=9, color=INK2)
    ax.text(x0 + w / 2 + 0.02, r["a"] + 1.5, f"{r['a']}\nfine-tuned", ha="center", fontsize=9, color=SLATE, weight="bold")
    ax.text(x0, r["a"] + 13, f"+{r['diff']:.0f} points, 95% CI +{r['lo']:.0f} to +{r['hi']:.0f}\nwon {r['won']} tasks, lost {r['lost']}, McNemar {pfmt(r['p'])}", fontsize=8.8, ha="center", va="bottom", color=SLATE)
ax.set_xticks(xs, ["first answer only", "best of 8, simulator picks"]); ax.set_xlim(-0.6, 1.85); ax.set_ylim(0, 118); ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_ylabel("tasks solved"); ax.tick_params(axis="x", labelsize=10, colors=SLATE)
save(fig, "fireworks")

# ---------- 6. ladder ----------
LD = [("Base 9B,\none answer", sum(r["oks"][0] for r in bon_base), BASE), ("Fine-tuned,\none answer", sft9, KRAFT),
      ("+ simulator picks\nbest of 8", sum(any(r["oks"]) for r in bon8), CLAY), ("+ best of 32", sum(any(r["oks"]) for r in bon32), CLAY),
      ("+ repair\n(8 samples)", sum(rep.values()), CLAY), ("+ repair v2,\n600 tests", o2[-1], CLAYD)]
fig, ax = frame("Each piece of the system adds tasks solved",
                "Tasks solved out of 100 as each idea is added, Qwen3.5-9B pipeline, same held-out tasks.",
                "Sources: runs/bon_base_n8.json, runs/river_sft_q9b_v3.json, runs/bon_q9b_v3_n8.json, runs/bon_q9b_v3_n32.json, runs/repair_v3_n8_stab.json, runs/trace_v3_repair2.json.",
                size=(9, 5.2), rect=(0.08, 0.17, 0.86, 0.60))
x = np.arange(len(LD)); vals = [v for _, v, _ in LD]
ax.bar(x, vals, 0.6, color=[c for *_, c in LD])
for i, v in enumerate(vals):
    ax.text(i, v + 1.5, str(v), ha="center", fontsize=11, weight="bold", color=SLATE)
    if i: ax.text(i, v - 4, f"+{v - vals[i-1]}", ha="center", va="top", fontsize=8.6, color=IVORY)
ax.axhline(89, color=SKY, lw=1, ls=(0, (3, 3))); ax.text(-0.42, 90.5, "GPT-6.1 Terra: 89", color=SKY, fontsize=8.4)
ax.set_xticks(x, [l for l, _, _ in LD]); ax.set_ylim(0, 108); ax.set_yticks([0, 25, 50, 75, 100])
ax.yaxis.grid(True); ax.set_axisbelow(True); ax.set_ylabel("tasks solved"); ax.tick_params(axis="x", labelsize=8.8)
save(fig, "ladder")

# ---------- 7. all comparisons: estimate, CI, McNemar ----------
fig, ax = frame("Every gain we claim survives both significance tests",
                "Difference in tasks solved with 95% paired bootstrap interval (10,000 resamples). Right: disagreements won and lost, exact McNemar.",
                "Filled dots: interval excludes zero and McNemar p < 0.05. Hollow dots: no detectable difference. With 100 tasks, gaps under about 5 points are not detectable.",
                size=(10, 5.6), rect=(0.33, 0.14, 0.42, 0.66))
yy = np.arange(len(res))[::-1]
for yi, r in zip(yy, res):
    sig = (r["lo"] > 0 or r["hi"] < 0) and r["p"] < 0.05; col = CLAY if sig else INK3
    ax.plot([r["lo"], r["hi"]], [yi, yi], color=col, lw=2.2, solid_capstyle="round")
    ax.plot(r["diff"], yi, "o", ms=7, color=col if sig else IVORY, mec=col, mew=1.6)
    ax.text(1.04, yi, f"{r['diff']:+.0f}  [{r['lo']:+.0f}, {r['hi']:+.0f}]", transform=ax.get_yaxis_transform(), va="center", fontsize=8.8, color=SLATE if sig else INK2)
    ax.text(1.30, yi, f"{r['won']} / {r['lost']}", transform=ax.get_yaxis_transform(), va="center", fontsize=8.8, color=SLATE if sig else INK2)
    ax.text(1.44, yi, pfmt(r["p"]), transform=ax.get_yaxis_transform(), va="center", fontsize=8.8, color=SLATE if sig else INK2)
for xx, h in ((1.04, "points, 95% CI"), (1.30, "won / lost"), (1.44, "McNemar")):
    ax.text(xx, len(res) - 0.25, h, transform=ax.get_yaxis_transform(), fontsize=8.2, color=INK3)
ax.axvline(0, color=INK3, lw=0.8); ax.set_yticks(yy, [r["label"] for r in res]); ax.set_xlim(-12, 62); ax.set_ylim(-0.6, len(res) - 0.1)
ax.spines["left"].set_visible(False); ax.xaxis.grid(True); ax.set_axisbelow(True); ax.set_xlabel("difference in tasks solved (points)")
ax.tick_params(axis="y", labelsize=8.8, colors=SLATE)
save(fig, "significance")

# ---------- 8. bootstrap distributions ----------
fig = plt.figure(figsize=(10, 4.6))
fig.text(0.04, 0.94, "Would the gap survive a different set of 100 tasks?", family=SERIF, fontsize=16, va="top")
fig.text(0.04, 0.865, "Each histogram is 10,000 resampled test sets. Lines mark the middle 95%.", fontsize=10, color=INK2, va="top")
fig.text(0.04, 0.035, "Resampling unit: individual tasks (10 categories are too few to resample as clusters). Script: stats/figures.py.", fontsize=7.8, color=INK3)
for i, (r, col, note) in enumerate([(res[3], CLAY, "ahead in {s:.0f}% of resamples: a real gain"), (res[7], SKY, "ahead in {s:.0f}% of resamples: a tie")]):
    ax = fig.add_axes((0.06 + i * 0.48, 0.17, 0.42, 0.56))
    ax.hist(r["boots"], bins=np.arange(-20, 75, 1), color=col, alpha=0.85)
    for q in (r["lo"], r["hi"]): ax.axvline(q, color=SLATE, lw=1)
    ax.axvline(0, ymax=0.68, color=INK3, lw=0.8, ls=(0, (3, 3)))
    ax.set_title(r["label"], loc="left", fontsize=10, color=SLATE, pad=6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("difference in tasks solved (points)")
    ax.text(0.02 if i == 0 else 0.98, 0.95, f"{r['diff']:+.0f} point{'' if abs(round(r['diff'])) == 1 else 's'}\n95% CI {r['lo']:+.0f} to {r['hi']:+.0f}\n" + note.format(s=(r["boots"] > 0).mean() * 100),
            transform=ax.transAxes, ha="left" if i == 0 else "right", va="top", fontsize=8.6, color=SLATE)
save(fig, "bootstrap")

# ---------- 9. robustness ----------
m = rob["mean"]; dc = rob["ours_minus_cold_spec_frac"]; dg = rob["ours_minus_gpt_spec_frac"]
RB = [("Optimizer only", m["cold"]["spec_frac"] * 100, INK3), ("Joint Tuner", m["ours"]["spec_frac"] * 100, CLAY), ("GPT-6.1 Terra", m["gpt"]["spec_frac"] * 100, SKY)]
fig, ax = frame("Under model mismatch, our gains beat the optimizer's and trail GPT's",
                f"Share of perturbed plants (every coefficient scaled by 0.8 to 1.2, 50 per task) that still meet the full spec. {rob['n_common']} tasks all three solve.",
                "All three keep 100% of plants stable. Intervals are paired bootstrap 95%. Source: control/robustness.py, runs/robustness.json.",
                size=(9, 4.6), rect=(0.2, 0.17, 0.74, 0.56))
yb = np.arange(3)[::-1]
ax.barh(yb, [v for _, v, _ in RB], 0.58, color=[c for *_, c in RB])
for yi, (lab, v, c) in zip(yb, RB): ax.text(v + 0.8, yi, f"{v:.1f}%", va="center", fontsize=10, weight="bold" if c == CLAY else "normal")
ax.set_yticks(yb, [l for l, _, _ in RB]); ax.set_xlim(0, 135); ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
ax.spines["left"].set_visible(False); ax.xaxis.grid(True); ax.set_axisbelow(True); ax.tick_params(axis="y", labelsize=10, colors=SLATE)
ax.text(m["ours"]["spec_frac"] * 100 + 11, 1.0, f"+{dc['mean']*100:.1f} vs optimizer\n[{dc['ci95'][0]*100:+.1f}, {dc['ci95'][1]*100:+.1f}]", fontsize=8.4, va="center", color=SLATE)
ax.text(m["gpt"]["spec_frac"] * 100 + 11, 0.0, f"Joint Tuner is {dg['mean']*100:.1f}\n[{dg['ci95'][0]*100:.1f}, {dg['ci95'][1]*100:.1f}]", fontsize=8.4, va="center", color=INK2)
save(fig, "robustness")

# ---------- 10. flywheel ----------
fig, _ = frame("Every tuning job makes the next one cheaper", "Each step test is a labeled example. Feeding them back into Fireworks training shrinks the number of tests the next joint needs.",
               "RL with the simulator as reward follows PI-GRPO (arXiv 2607.26594) and SciDesignBench RLSF (arXiv 2603.12724). Not yet run in this project.", size=(10, 5.4), rect=None)
ax = fig.add_axes((0.03, 0.09, 0.94, 0.74)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
N = [(0.32, 0.80, "Propose", "open model on Fireworks\nwrites candidate gains", CLAY), (0.56, 0.47, "Test", "simulator first,\nthen the real joint", OAT),
     (0.32, 0.14, "Log", "gains, measured response,\npass or fail", OAT), (0.10, 0.47, "Retrain", "fine-tune on successes,\nthen RL on simulator reward", OAT)]
bw, bh = 0.20, 0.22
for x0, y0, t, b, fc in N:
    ax.add_patch(FancyBboxPatch((x0 - bw / 2, y0 - bh / 2), bw, bh, boxstyle="round,pad=0,rounding_size=0.02", fc=fc, ec=CLAY if fc == CLAY else INK3, lw=1))
    ax.text(x0, y0 + 0.055, t, family=SERIF, fontsize=12.5, ha="center", va="center", color=IVORY if fc == CLAY else SLATE)
    ax.text(x0, y0 + 0.0, b, fontsize=7.8, ha="center", va="top", color=IVORY if fc == CLAY else INK2, linespacing=1.35)
for p0, p1 in [((0.42, 0.80), (0.56, 0.58)), ((0.56, 0.36), (0.42, 0.14)), ((0.22, 0.14), (0.10, 0.36)), ((0.10, 0.58), (0.22, 0.80))]:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, color=INK2, lw=1.1, connectionstyle="arc3,rad=-0.3", shrinkA=3, shrinkB=3))
ax.text(0.71, 0.88, "Where it scales next", family=SERIF, fontsize=12.5, color=SLATE, va="top")
for i, (h, b) in enumerate([("More robot types", "arms, humanoids, drones, actuators:\ngenerate tasks from their dynamics"),
                            ("Harder loops", "multi-joint, cascaded loops, gain schedules"),
                            ("Field data", "real step tests close the gap\nbetween simulator and hardware"),
                            ("Higher first-answer accuracy", "RL lifts the 45 to 50% first answer\ntoward the 90% best-of-N ceiling")]):
    ax.text(0.71, 0.76 - i * 0.17, h, fontsize=9.6, color=SLATE, weight="bold", va="top")
    ax.text(0.71, 0.71 - i * 0.17, b, fontsize=8.6, color=INK2, va="top", wrap=True)
save(fig, "flywheel")
for r in res: print(f"{r['label']:48s} {r['a']:3d} vs {r['b']:3d}  {r['won']}/{r['lost']}  {pfmt(r['p'])}  {r['diff']:+.0f} [{r['lo']:+.0f},{r['hi']:+.0f}]")
