# Paper insights → how we improve (2026-11-02 morning)

| Paper | What they did | What we take |
|---|---|---|
| ControlAgent (arXiv 2410.19811), source of ControlEval | GPT agent loop: design → Python evaluates → feedback with structured memory (past gains + metrics) → redesign, up to 20 iterations for unstable plants. ~100% on most categories incl. 2nd-order unstable. Zero-shot LLMs 2–56%. | Iterative simulator feedback is what cracks the unstable plants. One-shot sampling can't. |
| PID tuning with LLM agents + PI-GRPO (arXiv 2607.26594) | Qwen3-0.6B. SFT on 40k sim-verified targets, half with 1–3 "feedback turns" (failed design → corrected design). Then GRPO with gated physics reward (stability gate → pass/fail → IAE). Pass@1 86.5 → 94.0, Pass@8 95. FOPDT/SOPDT only. | (1) Train on correction turns, not only first answers. (2) RL lifts pass@1 toward pass@k. Our gap is huge (pass@1 44 vs pass@32 89), so RL has room. Fireworks serverless supports RL in the same session. |
| CoDyControlBench + reasoning distillation (arXiv 2608.07004) | GPT-5.5 94.8% on 132 configs. A 1.5B distilled on GPT reasoning traces beats answer-only distillation and transfers to a real robot arm. | Distill frontier traces (e.g. GPT-6.1 Terra) as a possible SFT upgrade. Real-hardware demo is a strong story. |
| LLM calibration agent (arXiv 2610.22379) | LLM gives a strong first guess; its advantage fades after ~15 evaluations. Hybrid: LLM warm-start → classical optimizer reaches target in 9 runs vs NM 17, BO >60. | **LLM proposes, numerical optimizer polishes.** Implemented as propose → simulate → repair. |
| SciDesignBench RLSF (arXiv 2603.12724) | RL from simulator feedback on an 8B model: +8–17 pts single-turn. | Same recipe as PI-GRPO; confirms direction. |
| Refining over resampling (arXiv 2608.05643), Frontier-Eng (2604.12290) | Depth (refining) beats pure width (more samples) at a fixed budget. | Repair loop > bigger N. |

## Our innovation: Propose → Simulate → Repair (control/propose_repair.py)
1. Fine-tuned small model proposes N PID candidates (one call, ~1 s).
2. Simulator scores all of them (ms).
3. If none pass: Nelder-Mead in log-gain space, warm-started from the top-3 candidates, with a fixed budget and a smooth stability penalty.
Results on the same 100 ControlEval tasks (v3 9B on River, N=8):
- sample only 71–79 → +repair **88–90**/100, p50 5–6 s. Same as best-of-32 (89) or better, with 4x fewer LLM samples.
- Unstable categories still weak → testing derivative-heavy seeds + unbounded stability penalty.
Next: RL (GRPO) on Fireworks serverless with the judge as reward; correction-turn SFT data.

## Ablation: optimizer alone vs model + optimizer (same 100 tasks)
| System | Correct | Median simulator calls | p90 calls |
|---|---|---|---|
| Nelder-Mead cold start, 3 starts × 150 | 84 | 98.5 | 450 |
| Ours: 9B proposes 8 + repair | 90 | 8 | 527 |
Ours never loses a task the cold optimizer solves (6–0, McNemar p=0.031). The model solves 79/100 with only its 8 proposals checked.
