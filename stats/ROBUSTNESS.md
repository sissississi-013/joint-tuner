# Robustness to plant perturbation

Each method's final passing gains, re-evaluated on 50 perturbed plants per task (every num/den coefficient except the leading den coefficient, and tau, scaled by an independent U[0.8, 1.2] factor; spec unchanged). The perturbed plants are identical across methods (seed = crc32(uid)). Script: `control/robustness.py`; raw data: `runs/robustness.json`. All nominal gains passed again when re-checked (90/93/89).

## Overall (each method's own solved tasks)

| Method | Tasks | Still stable | Still meets full spec |
|---|---|---|---|
| Ours (Joint Tuner) | 90 | 100.0% | 83.3% |
| Optimizer only | 93 | 100.0% | 68.3% |
| GPT-6.1 Terra | 89 | 100.0% | 92.9% |

## Per category: mean fraction still meeting spec (n tasks)

| Category | Ours | Optimizer only | GPT-6.1 Terra |
|---|---|---|---|
| first_order_stable_fast | 88.6% (10) | 82.0% (10) | 91.8% (10) |
| first_order_stable_moderate | 94.0% (10) | 60.0% (10) | 96.8% (10) |
| first_order_stable_slow | 96.6% (10) | 74.4% (10) | 99.8% (10) |
| first_order_unstable | 87.2% (8) | 79.6% (9) | 93.3% (6) |
| first_order_w_delay | 82.8% (10) | 70.2% (10) | 93.0% (10) |
| higher_order | 60.6% (10) | 81.6% (10) | 91.8% (10) |
| second_order_stable_fast | 85.4% (10) | 38.8% (10) | 99.3% (9) |
| second_order_stable_moderate | 75.4% (10) | 64.2% (10) | 94.0% (10) |
| second_order_stable_slow | 83.8% (10) | 68.4% (10) | 88.8% (10) |
| second_order_unstable | 65.0% (2) | 59.0% (4) | 62.5% (4) |

Stability was 100% for every method in every category.

## Paired comparison: the 87 tasks all three methods solved

| Method | Still meets spec |
|---|---|
| Ours (Joint Tuner) | 82.8% |
| Optimizer only | 68.5% |
| GPT-6.1 Terra | 94.1% |

| Difference (spec retention) | Mean | 95% bootstrap CI (10k resamples over tasks) |
|---|---|---|
| Ours − Optimizer only | +14.3 pp | [+7.5, +21.3] pp |
| Ours − GPT-6.1 Terra | -11.3 pp | [-16.5, -6.2] pp |

## Takeaways

No gains from any method went unstable under ±20% parameter error, so stability does not separate the methods. Full-spec retention does. On paired tasks the Joint Tuner's gains keep meeting spec 83% of the time versus 69% for optimizer only, a significant +14 pp. GPT-6.1 Terra's gains are more robust than ours, at 94% (−11 pp, with a CI that excludes zero). The likely cause is that both search-based tuners stop at the first gains that pass, which tend to sit at the edge of the spec, while the LLM picks more conservative gains. Our weakest category is higher_order (61% vs 92% for GPT). The second_order_unstable rows cover only 2–4 tasks, so they are noisy.
