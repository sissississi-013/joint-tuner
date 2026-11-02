"""PID-tuner SFT on Fireworks Serverless Training + before/after eval on ControlEval (same 100 tasks).
Usage: python control/fw_sft.py STEPS BATCH EVAL_PER_CAT TAG   (env: BASE, TOK, RENDERER, LR, RANK, TRAIN, N_BEST, SKIP_BEFORE)
Writes runs/fw_sft_{TAG}.json with per-task rows (before/after greedy + best-of-N), latency, token cost."""
import os, sys, json, random, time, concurrent.futures as cf
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")
from fireworks.training import sdk
from fireworks.training.sdk import FiretitanServiceClient, FireworksClient
from training.renderer import TrainOnWhat, get_renderer, get_text_content
from training.renderer.tokenizer import get_tokenizer
from judge import load_tasks, evaluate, task_prompt
from fw import extract_json

STEPS, BATCH, PER, TAG = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
import re
CK = lambda s: re.sub(r"[^a-z0-9-]", "-", s.lower()).strip("-")[:17].strip("-")
BASE = os.environ.get("BASE", "accounts/fireworks/models/qwen3p6-35b-a3b")
TOK = os.environ.get("TOK", "Qwen/Qwen3.6-35B-A3B"); REND = os.environ.get("RENDERER", "qwen3_6_disable_thinking")
LR = float(os.environ.get("LR", "1e-4")); RANK = int(os.environ.get("RANK", "16")); NB = int(os.environ.get("N_BEST", "8"))
TRAIN = os.environ.get("TRAIN", "data/pid/train_v3.jsonl"); PRICE = dict(prefill=0.45, sample=1.335, train=2.14)  # qwen3p6-35b-a3b $/1M
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
tok = get_tokenizer(TOK); rend = get_renderer(REND, tok); STOP = rend.get_stop_sequences()
cost = defaultdict(float)

def datum(msgs):
    mi, w = rend.build_supervised_example(msgs, train_on_what=TrainOnWhat.LAST_ASSISTANT_TURN)
    t = [int(x) for x in (mi.to_ints() if hasattr(mi, "to_ints") else mi)]; w = [float(x) for x in (w.tolist() if hasattr(w, "tolist") else w)]
    return sdk.Datum(model_input=sdk.ModelInput.from_ints(t[:-1]), loss_fn_inputs={"target_tokens": t[1:], "weights": w[1:]}), len(t)

by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
EV = [t for c in sorted(by) for t in by[c][:PER]]

def eval_snapshot(svc, label, model_path=None, base_model=None):
    smp = svc.create_sampling_client(model_path=model_path, tokenizer=tok) if model_path else svc.create_sampling_client(base_model=base_model, tokenizer=tok)
    def one(t):
        p = rend.build_generation_prompt([{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}])
        out = {}
        for mode, n, temp in (("greedy", 1, 0.0), ("bon", NB, 0.8)):
            for a in range(5):
                try:
                    t0 = time.time(); r = smp.sample(prompt=p, num_samples=n, sampling_params=sdk.SamplingParams(max_tokens=60, temperature=temp, stop=STOP))
                    r = r.result() if hasattr(r, "result") else r; lat = time.time() - t0; break
                except Exception as e:
                    err = str(e)[:200]; time.sleep(3 * (a + 1))
            else:
                out[mode] = {"oks": [False] * n, "latency_s": None, "error": err}; continue
            oks, gains = [], []
            for s in r.sequences:
                toks = list(s.tokens); cost["sample_tok"] += len(toks)
                j = extract_json(get_text_content(rend.parse_response(toks)[0])) or {}
                ok = bool(j) and bool(evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd"))["success"]); oks.append(ok); gains.append(j)
            cost["prefill_tok"] += int(p.length)
            out[mode] = {"oks": oks, "gains": gains[:1], "latency_s": lat}
        return {"uid": t["uid"], "category": t["category"], **out}
    with cf.ThreadPoolExecutor(8) as ex: rows = list(ex.map(one, EV))
    smp.close()
    g = sum(r["greedy"]["oks"][0] for r in rows); b = sum(any(r["bon"]["oks"]) for r in rows)
    lats = sorted(r["greedy"]["latency_s"] for r in rows if r["greedy"]["latency_s"])
    cats = defaultdict(list)
    for r in rows: cats[r["category"]].append(r["greedy"]["oks"][0])
    print(f"[{label}] greedy {g}/{len(rows)}  best-of-{NB} {b}/{len(rows)}  p50 latency {lats[len(lats)//2]:.2f}s | " +
          " ".join(f"{c.replace('second_order','so').replace('first_order','fo')}={sum(v)}" for c, v in sorted(cats.items())), flush=True)
    return rows

rows = [json.loads(l) for l in open(TRAIN)]; random.Random(0).shuffle(rows)
data = [datum(r["messages"]) for r in rows]
print(f"{len(data)} training datums, base={BASE}", flush=True)

svc = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/training/v1/serverless")
tc = svc.create_lora_training_client(base_model=BASE, rank=RANK)
print("session", svc.training_session_id, "run", tc.run_id, flush=True)
result = {"base": BASE, "renderer": REND, "steps": STEPS, "batch": BATCH, "lr": LR, "rank": RANK, "train": TRAIN, "n_eval": len(EV)}
try:
    if not os.environ.get("SKIP_BEFORE"):
        result["before"] = eval_snapshot(svc, "before (base)", base_model=BASE)
    adam = sdk.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
    i = 0; t0 = time.time()
    for step in range(1, STEPS + 1):
        batch = data[i:i + BATCH]; i += BATCH
        if i >= len(data): i = 0
        fb = tc.forward_backward([d for d, _ in batch], "cross_entropy").result(); tc.optim_step(adam).result()
        cost["train_tok"] += sum(n for _, n in batch)
        if step % 100 == 0 and step < STEPS:
            try: result.setdefault("mid_checkpoints", []).append(tc.save_weights_for_sampler(CK(f"s{step}-{TAG}")).result().path); print("saved", result["mid_checkpoints"][-1], flush=True)
            except Exception as e: print("mid save failed", str(e)[:200], flush=True)
        if step == 1 or step % 25 == 0:
            m = getattr(fb, "metrics", {}) or {}; ls = m.get("loss:sum"); nt = m.get("response_tokens") or m.get("num_loss_tokens") or 1
            print(f"step {step} loss {float(ls)/max(float(nt),1):.4f} ({time.time()-t0:.0f}s)" if ls is not None else f"step {step}", flush=True)
    path = tc.save_weights_for_sampler(CK(f"pid-{TAG}")).result().path; result["checkpoint"] = path; print("checkpoint", path, flush=True)
    result["after"] = eval_snapshot(svc, "after (tuned)", model_path=path)
    try:
        fwc = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
        ck = [r for r in fwc.list_training_session_checkpoints(svc.training_session_name) if r.get("promotable")]
        ck.sort(key=lambda r: r["createTime"])
        m = fwc.promote_session_checkpoint(name=ck[-1]["name"], output_model_id=re.sub(r"[^a-z0-9-]", "-", f"pid-tuner-{TAG}".lower())[:63], base_model=BASE)
        result["promoted_model"] = getattr(m, "name", str(m)); print("promoted", result["promoted_model"], flush=True)
    except Exception as e:
        result["promote_error"] = str(e)[:300]; print("promote failed:", str(e)[:300], flush=True)
finally:
    result["tokens"] = dict(cost)
    result["est_cost_usd"] = round(cost["prefill_tok"] / 1e6 * PRICE["prefill"] + cost["sample_tok"] / 1e6 * PRICE["sample"] + cost["train_tok"] / 1e6 * PRICE["train"], 3)
    json.dump(result, open(f"runs/fw_sft_{TAG}.json", "w"), indent=1)
    print("estimated cost $", result["est_cost_usd"], dict(cost), flush=True)
    import httpx
    if not result.get("promoted_model"):
        print("NOT deleting session (no promoted model):", svc.training_session_name, flush=True); sys.exit(0)
    r = httpx.delete(f"https://api.fireworks.ai/v1/{svc.training_session_name}", headers={"Authorization": "Bearer " + os.environ["FIREWORKS_API_KEY"]})
    print("session deleted", r.status_code, flush=True)
