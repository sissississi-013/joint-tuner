# Joint Tuner

**A small open model that tunes PID controllers in a handful of physical tests.**

![Tuning race in MuJoCo: optimizer only vs GPT-6.1 Terra vs Joint Tuner on a wrist joint](media/mujoco_wrist.gif)

*The same wrist joint, rendered in MuJoCo, with each tuner's logged gains replayed at 4× speed. The optimizer needs 493 step tests. GPT-6.1 Terra thinks for 23 s, then succeeds on its first test. Joint Tuner succeeds on its 5th test after about 1 s of model time. Videos: [wrist joint](media/mujoco_wrist.mp4), [unstable joint](media/mujoco_pendulum.mp4).*

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

1. **Propose.** A fine-tuned open model writes 8 candidate PID gain sets in about 1 second.
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
- The optimizer catches up only with about 600 tests.
- Source: `control/budget_trace.py` → `runs/trace_v3.json`.

### Fine-tuning on Fireworks (Qwen3.6-35B-A3B, LoRA r=16, 300 steps × 32)

| | Greedy, first answer | Best-of-8, simulator selects |
|---|---|---|
| Base model | 20 | 53 |
| **Fine-tuned** (`pid-tuner-q36v3`) | **50** | **81** |

- Training cost: $5.11 on the Fireworks serverless training API.
- Median latency: 1.35 s.
- Source: `control/fw_sft.py` → `runs/fw_sft_q36v3.json`.

### Against frontier and open baselines (same 100 tasks)

| System | Solved | Median latency | Notes |
|---|---|---|---|
| GPT-6.1 Terra | 89 | 18.9 s | about $0.008 per task |
| **Joint Tuner, best-of-32** | **89** | 7.3 s | ties GPT (McNemar p = 1.0, 4 vs 4 discordant) |
| **Joint Tuner, 8 samples + repair** | **90** | 6.3 s | |
| Optimizer only, matched budget (3 × 150) | 84 | 9.6 s | Joint Tuner wins 6–0 on discordant tasks, McNemar p = 0.031 |
| Kimi K3, no thinking | 22 | | |
| Qwen3.8 27B | 11 | | |
| Nemotron 3.5 Super | 1 | | |
| Base Qwen3.5-9B | 8 (@1) / 41 (@8) | | |

The statistical tests are exact McNemar on discordant pairs plus a paired bootstrap 95% CI with 10,000 resamples (`stats/significance.py`).

## What lost (reported honestly)

- **Greedy single answer.** Only 45–50%. The simulator check is what makes the system reliable.
- **Bigger N alone.** Best-of-32 reaches 89. Best-of-8 plus repair reaches the same with 4× fewer model samples.
- **Repair with a flat instability penalty.** It gave the optimizer no direction on unstable plants. A smooth pole-based penalty plus derivative-heavy seeds moved unstable tasks from 8/20 to 12/20, but cost a median of 165 s per task.
- **Second-order unstable plants** are still our weakest category (2/10, versus 4/10 for GPT-6.1 Terra).
- **Kimi K3 with thinking** truncated at 16k tokens on 60/100 tasks. We re-ran those at 100k tokens so the baseline is not unfairly weakened (`control/rerun_truncated.py`).
- **Fireworks SFT run 1** (400 steps) was lost: the checkpoint name contained an underscore, which the API rejects. The script now sanitizes names, saves every 100 steps, and only deletes the session after the model is promoted.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install numpy scipy control python-dotenv openai "fireworks-ai[training]>=1.2.29"
cp .env.example .env          # add your own keys
./fetch_controleval.sh        # ControlEval is fetched from upstream, not redistributed

python control/gen_synth.py                                   # reverse-generated, simulator-verified SFT data
python control/fw_sft.py 300 32 10 q36v3                      # Fireworks serverless LoRA SFT + before/after eval
python control/propose_repair.py <BASE> <CKPT> 8 1.0 v3       # propose → simulate → repair
python control/coldstart_ablation.py 150 3                    # optimizer-only ablation
python control/budget_trace.py <CKPT> 8 v3                    # main figure data
python control/closed_baseline.py openai gpt-6.1-terra 10     # frontier baseline
python stats/significance.py                                  # McNemar + bootstrap
python demo/build_race.py && open demo/race.html              # web demo
python demo/mujoco_race.py second_order_stable_fast/6 media/mujoco_wrist.mp4   # MuJoCo render (pip install mujoco imageio[ffmpeg] pillow)
```

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
