# Joint Tuner

**A small open model that tunes PID controllers in a handful of physical tests.**

![Tuning race in MuJoCo: optimizer only vs GPT-6.1 Terra vs Joint Tuner on a wrist joint](media/mujoco_wrist.gif)

*The same wrist joint, rendered in MuJoCo, with each tuner's logged gains replayed at 4× speed. The optimizer needs 493 step tests. GPT-6.1 Terra thinks for 23 s, then succeeds on its first test. Joint Tuner succeeds on its 5th test. The clip shows the 9B model's proposals with a 1 s model wait, the Fireworks model's median latency (1.35 s). The logged River call for this task took 9 s. Videos: [wrist joint](media/mujoco_wrist.mp4), [unstable joint](media/mujoco_pendulum.mp4).*

## Where this runs on real hardware

In each scene, the **orange unit** is simulated in MuJoCo with the exact dynamics of a held-out task. Inertia, damping, stiffness and motor gain come straight from its transfer function. The drone adds rotor lag. The unit is driven by Joint Tuner's PID gains, and every scene meets its spec in both the judge and MuJoCo. The rest of each body is fixed for clarity.

### Actuator: gearbox output shaft
2nd-order stable plant, task `second_order_stable_fast/6`, played in real time. [MP4](media/actuator.mp4)

![Actuator: the gearbox output shaft is highlighted](media/actuator.gif)

### Robot arm: elbow joint
2nd-order stable plant with a slow spec, task `second_order_stable_moderate/9`, played at 37× speed. [MP4](media/robot_arm_elbow.mp4)

![Robot arm: the elbow joint and forearm are highlighted](media/robot_arm_elbow.gif)

### Humanoid: ankle balance
2nd-order **unstable** plant, task `second_order_unstable/7`, played in real time. The ankle balances the whole body as an inverted pendulum. [MP4](media/humanoid_ankle.mp4)

![Humanoid: the ankle pitch joint is highlighted](media/humanoid_ankle.gif)

### Drone: roll axis on a test stand
Plant K/(s²(s+a)), with rotor lag. Task `drone/4` is held-out synthetic. Played at 2× speed. [MP4](media/drone_roll.mp4)

![Drone: the roll axis is highlighted](media/drone_roll.gif)

The physics behind the mapping: [docs/physical-model.md](docs/physical-model.md).

Tuning a motor joint on real hardware means running step tests: the joint moves, you measure, you reset. Each test costs time, wear, and risk. Joint Tuner pairs a fine-tuned open model with a simulator check:

1. **Propose.** A fine-tuned open model writes 8 candidate PID gain sets in one call (1.35 s median on Fireworks).
2. **Simulate.** A control-theory judge checks each set against the spec: phase margin, gain margin, settling time, overshoot, and steady-state error.
3. **Repair.** If none pass, Nelder-Mead polishes the best candidates in log-gain space under a fixed test budget.

The production model (`pid-tuner-q36v3`, Qwen3.6-35B-A3B + LoRA) is trained and evaluated with the Fireworks serverless training API and promoted to a Fireworks model. We tried an on-demand deployment for live serving, but it failed at creation with a Fireworks internal error during the event, so all Fireworks numbers come from the training session's sampler. Pipeline experiments below were first run with a Qwen3.5-9B LoRA trained on River during development; rows say which model they use.

## Results

All results use the same 100 held-out tasks from ControlEval (arXiv 2410.19811): 10 categories × 10 tasks, scored by the same judge (`control/judge.py`).

### Main figure: tasks solved vs. step tests allowed

| Step tests allowed | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Joint Tuner** (9B River-trained, propose + repair) | **46** | **57** | **69** | **76** | **85** | **86** | **87** | **87** | **90** | 90 |
| Optimizer only (Nelder-Mead, no model) | 19 | 20 | 20 | 27 | 43 | 58 | 62 | 67 | 77 | 93 |
| GPT-6.1 Terra, one answer | 89 | | | | | | | | | |

- Joint Tuner leads the optimizer at every budget up to 256 tests. At 8 tests it solves 76 tasks to the optimizer's 27.
- The optimizer ties us only at 448 tests and passes us at about 460.
- **Repair v2** (`control/repair_v2.py`, offline replay of the same logged proposals) fixes the large-budget gap. When the model's starting points run out, it falls back to the optimizer's own restart schedule, after scaled seeds for unstable plants. Result: **93/100 at 600 tests, tied with optimizer only on the same 93 tasks**, with the curve at 1–256 tests unchanged. Caveats: the variant was chosen on these same 100 tasks, the +3 comes from 3 unstable tasks, and on 2 of them the fallback reaches the optimizer's solution later than the optimizer does. Source: `runs/trace_v3_repair2.json`.
- Source: `control/budget_trace.py` → `runs/trace_v3.json`.

### Is the difference real? (`stats/RESULTS.md`)

| Step tests allowed | Joint Tuner | Optimizer only | Difference, 95% CI | McNemar p |
|---|---|---|---|---|
| 1 | 46 | 19 | +27 pts [15, 39] | 1e-4 |
| 8 | 76 | 27 | +49 pts [38, 59] | 3e-13 |
| 256 | 90 | 77 | +13 pts [6, 21] | 0.002 |
| 600 | 90 | 93 | −3 pts [−7, 0] | 0.25, not significant |

- Every budget from 1 to 256 is significant after a Holm correction.
- **Tests to first success** (on the 90 tasks both methods solve): a median of **1** for Joint Tuner (IQR 1–3.75) vs **19.5** for the optimizer (IQR 6–166). Wilcoxon p ≈ 1e-10.
- **Area under the curve** for budgets up to 256: 0.77 vs 0.43, difference +0.34 [0.26, 0.42].
- **Against GPT-6.1 Terra:** 90 vs 89, McNemar p = 1.0. Not significantly different, and not a claim that we beat it.

### Fine-tuning on Fireworks (Qwen3.6-35B-A3B, LoRA r=16, 300 steps × 32)

| | Greedy, first answer | Best-of-8, simulator selects |
|---|---|---|
| Base model | 20 | 53 |
| **Fine-tuned** (`pid-tuner-q36v3`) | **50** | **81** |

- Training cost: $5.11 on the Fireworks serverless training API.
- Median latency: 1.35 s.
- Source: `control/fw_sft.py`. The fine-tuned numbers are in `runs/fw_sft_q36v3.json`. The base-model numbers are in `runs/fw_sft_q36v3_before_eval.json`, the first run's pre-training evaluation of the same base model on the same 100 tasks.

### Against frontier and open baselines (same 100 tasks)

| System | Solved | Median latency | Notes |
|---|---|---|---|
| GPT-6.1 Terra | 89 | 18.7 s | about $0.008 per task |
| **Joint Tuner, best-of-32** | **89** | 4.9 s | statistically indistinguishable from GPT (McNemar p = 1.0, 4 vs 4 discordant; low power: only gaps of 5–6+ pts are detectable at n=100) |
| **Joint Tuner, 8 samples + repair** | **90** | 6.3 s | |
| Optimizer only, matched budget (3 × 150) | 84 | 9.4 s | Joint Tuner wins 6–0 on discordant tasks, McNemar p = 0.031 |
| Kimi K3 with thinking (open model, up to 100k tokens) | 90 | 222 s | ties us on accuracy at about 35x the model time (`runs/pid_fireworks_kimi-k3_think_full100.json`) |
| Kimi K3, no thinking | 22 | | |
| Qwen3.8 27B | 11 | | |
| Nemotron 3.5 Super | 1 | | |
| Base Qwen3.5-9B | 8 (@1) / 41 (@8) | | |

The statistical tests are exact McNemar on discordant pairs plus a paired bootstrap 95% CI with 10,000 resamples (`stats/budget_stats.py` → `stats/RESULTS.md`; earlier 50-task comparisons are in `stats/significance.py`).

## Robustness to model mismatch (`stats/ROBUSTNESS.md`)

Real joints never match their model. We perturbed every plant coefficient by ±20%, with 50 random plants per task. Then we re-checked each method's final gains. The comparison uses the 87 tasks that all three methods solve.

| Gains from | Still stable | Still meets full spec |
|---|---|---|
| Optimizer only | 100% | 68.5% |
| **Joint Tuner** | 100% | **82.8%**: +14.3 pts vs optimizer [7.5, 21.3] |
| GPT-6.1 Terra | 100% | 94.1%: Joint Tuner is −11.3 pts [−16.5, −6.2] |

GPT's gains are significantly more robust than ours, and we report that as a weakness. One likely reason is that our repair stops at the first passing gain set, which often sits at the edge of the spec. Picking the most robust of the passing candidates is a next step.

## What lost (reported honestly)

- **Greedy single answer.** Only 45–50%. The simulator check is what makes the system reliable.
- **Bigger N alone.** Best-of-32 reaches 89. Best-of-8 plus repair reaches the same with 4× fewer model samples.
- **Repair with a flat instability penalty.** It gave the optimizer no direction on unstable plants. A smooth pole-based penalty plus derivative-heavy seeds moved unstable tasks from 10/20 to 12/20, but cost a median of 130 s per task.
- **Second-order unstable plants** are still our weakest category (1–2/10 depending on the run, versus 4/10 for GPT-6.1 Terra and 4/10 for the optimizer at 600 tests).
- **Kimi K3 with thinking** truncated at 16k tokens on 60/100 tasks. We re-ran those at 100k tokens so the baseline is not unfairly weakened (`control/rerun_truncated.py`). With full thinking it reaches 90/100, at a median of 222 s per task.
- **Fireworks SFT run 1** (400 steps) was lost: the checkpoint name contained an underscore, which the API rejects. The script now sanitizes names, saves every 100 steps, and only deletes the session after the model is promoted.

## Reproduce

Run everything from the repo root; scripts use relative paths (`control/`, `runs/`, `data/`).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
./fetch_controleval.sh        # ControlEval is fetched from upstream, not redistributed
```

**No API keys needed.** These check the published numbers from the logged results in `runs/`:

```bash
python stats/significance.py                                  # McNemar + bootstrap -> runs/significance.json
python stats/budget_stats.py                                  # budget-curve stats -> runs/budget_stats.json, stats/RESULTS.md
python control/coldstart_ablation.py 150 3                    # optimizer-only ablation, ~minutes on a laptop -> runs/coldstart_150x3.json
python control/gen_synth.py 7500 data/pid/train_new.jsonl 0   # reverse-generated, simulator-verified SFT data (N OUT SEED)
python demo/build_race.py && open demo/race.html              # web demo
python demo/mujoco_race.py second_order_stable_fast/6 media/mujoco_wrist.mp4   # MuJoCo race render
python demo/mujoco_units.py                                   # the four hardware scenes in media/
```

**Needs keys** (`cp .env.example .env` and fill in your own):

```bash
python control/fw_sft.py 300 32 10 q36v3                      # FIREWORKS_API_KEY: serverless LoRA SFT + before/after eval
python control/closed_baseline.py openai gpt-6.1-terra 10     # OPENAI_API_KEY: frontier baseline
FW_MODEL=accounts/<acct>/models/<model> python control/propose_repair.py Qwen/Qwen3.5-9B - 8 1.0 v3   # propose -> simulate -> repair
FW_MODEL=accounts/<acct>/models/<model> python control/budget_trace.py - 8 v3                         # main figure data
```

### What needs River

The 9B rows (`runs/river_sft_*`, `runs/bon_*`, `runs/repair_*`, `runs/trace_v3.json`) were sampled through River, a private training/sampling SDK (`river_client`) that is not public. Those raw outputs are all in `runs/`, and every downstream number (judge, stats, demos) recomputes from them without River.

- `control/river_sft.py`, `control/bestofn.py` and `control/eval_file.py` (any mode other than `chat`) need River. The first two exit with a message if it is missing.
- `control/propose_repair.py` and `control/budget_trace.py` run without River when `FW_MODEL` points at a Fireworks model. That reproduces the pipeline, not the exact 9B numbers.
- `control/gen_synth.py` was run with an unrecorded seed, so a regenerated file will differ from `data/pid/train_v3.jsonl` (which is included).

## Layout

- `control/judge.py`: the ControlEval-style spec checker, which is also the reward.
- `control/gen_synth.py`: training data. Labels come from sampled gains whose specs are derived backward, so every label is feasible.
- `control/propose_repair.py`: the propose → simulate → repair pipeline.
- `control/fw_sft.py`: Fireworks serverless training, evaluation and promotion.
- `demo/mujoco_units.py`: renders the four hardware scenes, with the tuned unit highlighted.
- `demo/mujoco_race.py`: renders the race in MuJoCo, with each plant built as a hinge joint (inertia, damping, stiffness, motor gear).
- `docs/physical-model.md`: transfer function ↔ joint physics, plus the category-to-hardware map for arms, humanoids and drones.
- `demo/race.html`: "How many tries does it take to tune a motor?" Three tuners race on the same motor.
- `docs/related-work.md`: the papers we built on and what we took from each.
- `runs/`: raw per-task results for every number above.

## Related work

ControlAgent (2410.19811), PI-GRPO PID tuning (2607.26594), CoDyControlBench (2608.07004), hybrid LLM warm-start calibration (2610.22379), and SciDesignBench RLSF (2603.12724). See `docs/related-work.md`.
