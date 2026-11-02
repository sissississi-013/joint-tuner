"""Paired statistics for Joint Tuner vs optimizer-only vs GPT-6.1 Terra on ControlEval (100 tasks).

Run: .venv/bin/python stats/budget_stats.py   (from repo root). Offline; reads runs/*.json only.
Outputs: runs/budget_stats.json, stats/RESULTS.md
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
N_BOOT, SEED = 10_000, 0


def load(name):
    return json.loads((RUNS / f"{name}.json").read_text())


trace = load("trace_v3")
budgets = trace["budgets"]
rows = sorted(trace["rows"], key=lambda r: r["uid"])
uids = [r["uid"] for r in rows]
cats = np.array([r["category"] for r in rows])
N = len(uids)


def by_uid(lst, fn):
    d = {x["uid"]: fn(x) for x in lst}
    missing = set(uids) - set(d)
    assert not missing, f"missing uids: {sorted(missing)[:5]}"
    return np.array([bool(d[u]) for u in uids])


gpt = by_uid(load("pid_closed_gpt-6.1-terra_10"), lambda x: x["success"])
repair = by_uid(load("repair_v3_n8_stab"), lambda x: x["sample_ok"] or x["repair_ok"])
bo32 = by_uid(load("bon_q9b_v3_n32"), lambda x: any(x["oks"]))
cold150 = by_uid(load("coldstart_150x3"), lambda x: x["success"])

INF = np.inf
ours_first = np.array([r["ours_first"] if r["ours_first"] is not None else INF for r in rows], float)
cold_first = np.array([r["cold_first"] if r["cold_first"] is not None else INF for r in rows], float)

rng = np.random.default_rng(SEED)
IDX = rng.integers(0, N, size=(N_BOOT, N))  # one shared resample matrix: paired, reproducible


def boot_ci(a, b):
    """Paired bootstrap 95% percentile CI of mean(a) - mean(b) (per-task values)."""
    d = np.asarray(a, float) - np.asarray(b, float)
    means = d[IDX].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def mcnemar(a, b):
    """Exact two-sided McNemar: binomial test on discordant pairs."""
    n10 = int(np.sum(a & ~b))  # a solves, b doesn't
    n01 = int(np.sum(~a & b))
    p = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p


def compare(a, b):
    n10, n01, p = mcnemar(a, b)
    diff, lo, hi = boot_ci(a, b)
    return dict(a=int(a.sum()), b=int(b.sum()), a_only=n10, b_only=n01, mcnemar_p=p,
                diff=diff, ci_lo=lo, ci_hi=hi)


out = {"n_tasks": N, "n_boot": N_BOOT, "seed": SEED}

# 1. per-budget
per_budget = []
for B in budgets:
    o, c = ours_first <= B, cold_first <= B
    per_budget.append(dict(budget=B, **compare(o, c)))
# sanity vs the stored curve
assert [r["a"] for r in per_budget] == trace["curve"]["ours"], "ours curve mismatch"
assert [r["b"] for r in per_budget] == trace["curve"]["cold"], "cold curve mismatch"
out["per_budget"] = per_budget

# 2. AUC of success vs log2(B), B<=256 (trapezoid, normalized to [0,1] = mean solve rate over log-budget)
auc_b = [B for B in budgets if B <= 256]
x = np.log2(auc_b)


def per_task_auc(first):
    s = np.stack([(first <= B).astype(float) for B in auc_b], axis=1)  # N x K
    return np.trapezoid(s, x, axis=1) / (x[-1] - x[0])


auc_o, auc_c = per_task_auc(ours_first), per_task_auc(cold_first)
d, lo, hi = boot_ci(auc_o, auc_c)
# permutation-free p: fraction of bootstrap diffs on the other side of 0 (two-sided), reported as a rough guide
bd = (auc_o - auc_c)[IDX].mean(axis=1)
p_boot = float(min(1.0, 2 * min((bd <= 0).mean(), (bd >= 0).mean())))
w_auc = wilcoxon(auc_o, auc_c, zero_method="wilcox")
out["auc"] = dict(budgets=auc_b, ours=float(auc_o.mean()), cold=float(auc_c.mean()), diff=d, ci_lo=lo, ci_hi=hi,
                  boot_p_two_sided=p_boot, wilcoxon_p=float(w_auc.pvalue))

# 3. tests-to-first-success on tasks both solve (at max budget)
both = np.isfinite(ours_first) & np.isfinite(cold_first)


def mi(v):
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return dict(median=float(med), q1=float(q1), q3=float(q3))


w = wilcoxon(ours_first[both], cold_first[both], zero_method="wilcox")
ratio = cold_first[both] / ours_first[both]
out["tests_to_first"] = dict(n_both=int(both.sum()), ours=mi(ours_first[both]), cold=mi(cold_first[both]),
                             ours_fewer=int(np.sum(ours_first[both] < cold_first[both])),
                             cold_fewer=int(np.sum(ours_first[both] > cold_first[both])),
                             ties=int(np.sum(ours_first[both] == cold_first[both])),
                             median_ratio_cold_over_ours=float(np.median(ratio)),
                             wilcoxon_stat=float(w.statistic), wilcoxon_p=float(w.pvalue))

# 4. vs GPT
out["vs_gpt"] = {"repair_vs_gpt": compare(repair, gpt), "bo32_vs_gpt": compare(bo32, gpt),
                 "trace_ours_full_vs_gpt": compare(ours_first <= budgets[-1], gpt),
                 "cold150x3_vs_gpt": compare(cold150, gpt)}
out["notes"] = {
    "repair_vs_trace_full_disagree": int(np.sum(repair != (ours_first <= budgets[-1]))),
    "cold150x3_vs_trace_cold_full_disagree": int(np.sum(cold150 != (cold_first <= budgets[-1]))),
}

# 5. per-category
cat_rows = []
for cat in sorted(set(cats)):
    m = cats == cat
    cat_rows.append(dict(category=cat, n=int(m.sum()),
                         ours8=int(np.sum(ours_first[m] <= 8)), cold8=int(np.sum(cold_first[m] <= 8)),
                         ours_full=int(np.sum(np.isfinite(ours_first[m]))),
                         cold_full=int(np.sum(np.isfinite(cold_first[m]))),
                         gpt=int(gpt[m].sum())))
out["per_category"] = cat_rows

(RUNS / "budget_stats.json").write_text(json.dumps(out, indent=2))


# ---------- markdown ----------
def fp(p):
    return "<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def pp(v):
    return f"{100 * v:+.1f}"


L = ["# ControlEval statistics: Joint Tuner vs optimizer-only vs GPT-6.1 Terra", "",
     f"100 tasks (10 categories x 10). All tests are paired on the same tasks. McNemar = exact two-sided binomial test "
     f"on discordant pairs. CIs = paired percentile bootstrap, {N_BOOT:,} resamples, seed {SEED}. "
     f"Generated by `stats/budget_stats.py` (raw numbers in `runs/budget_stats.json`).", "",
     "## 1. Solve rate vs simulator-test budget (Joint Tuner vs Nelder-Mead cold start)", "",
     "| Budget B | Ours | Cold | Ours-only | Cold-only | McNemar p | Diff (pp) | 95% CI (pp) |",
     "|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in per_budget:
    L.append(f"| {r['budget']} | {r['a']} | {r['b']} | {r['a_only']} | {r['b_only']} | {fp(r['mcnemar_p'])} | "
             f"{pp(r['diff'])} | [{pp(r['ci_lo'])}, {pp(r['ci_hi'])}] |")
a = out["auc"]
L += ["", "## 2. Area under success vs log2(budget), B = 1..256", "",
      "Normalized trapezoid area (0-1 scale = average solve rate across log-spaced budgets).", "",
      "| Ours | Cold | Diff | 95% CI | Wilcoxon p (per-task AUC) |", "|---:|---:|---:|---:|---:|",
      f"| {a['ours']:.3f} | {a['cold']:.3f} | {a['diff']:+.3f} | [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}] | {fp(a['wilcoxon_p'])} |"]
t = out["tests_to_first"]
L += ["", f"## 3. Simulator tests to first success (tasks both methods solve, n = {t['n_both']})", "",
      "| Method | Median | IQR |", "|---|---:|---:|",
      f"| Joint Tuner | {t['ours']['median']:g} | {t['ours']['q1']:g}-{t['ours']['q3']:g} |",
      f"| Optimizer only | {t['cold']['median']:g} | {t['cold']['q1']:g}-{t['cold']['q3']:g} |", "",
      f"Ours fewer tests on {t['ours_fewer']} tasks, cold fewer on {t['cold_fewer']}, ties {t['ties']}. "
      f"Median per-task ratio (cold/ours) = {t['median_ratio_cold_over_ours']:.1f}x. "
      f"Wilcoxon signed-rank p = {fp(t['wilcoxon_p'])}."]
L += ["", "## 4. Versus GPT-6.1 Terra (single answer, budget 1)", "",
      "| Comparison | A solved | GPT solved | A-only | GPT-only | McNemar p | Diff (pp) | 95% CI (pp) |",
      "|---|---:|---:|---:|---:|---:|---:|---:|"]
names = {"repair_vs_gpt": "Joint Tuner (8 samples + repair)", "bo32_vs_gpt": "Fine-tuned 9B best-of-32",
         "trace_ours_full_vs_gpt": "Joint Tuner trace @600", "cold150x3_vs_gpt": "Optimizer only (150x3 run)"}
for k, r in out["vs_gpt"].items():
    L.append(f"| {names[k]} | {r['a']} | {r['b']} | {r['a_only']} | {r['b_only']} | {fp(r['mcnemar_p'])} | "
             f"{pp(r['diff'])} | [{pp(r['ci_lo'])}, {pp(r['ci_hi'])}] |")
L += ["", "## 5. Per-category solves (out of 10)", "",
      f"'full' = trace max budget ({budgets[-1]} tests).", "",
      "| Category | Ours@8 | Cold@8 | Ours@full | Cold@full | GPT |", "|---|---:|---:|---:|---:|---:|"]
for r in cat_rows:
    L.append(f"| {r['category']} | {r['ours8']} | {r['cold8']} | {r['ours_full']} | {r['cold_full']} | {r['gpt']} |")
tot = {k: sum(r[k] for r in cat_rows) for k in ["ours8", "cold8", "ours_full", "cold_full", "gpt"]}
L.append(f"| **Total** | {tot['ours8']} | {tot['cold8']} | {tot['ours_full']} | {tot['cold_full']} | {tot['gpt']} |")

(ROOT / "stats" / "RESULTS.md").write_text("\n".join(L) + "\n")
print(json.dumps(out, indent=1))
