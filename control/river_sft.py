"""SFT a LoRA on River with synthetic PID tasks, then evaluate on ControlEval (held out).
Usage: python control/river_sft.py BASE_MODEL STEPS BATCH EVAL_PER_CAT [tag]"""
import os, sys, json, random, time, concurrent.futures as cf
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")
try:
    import river_client as river
    from river_client.renderers import get_renderer
except ImportError:
    sys.exit("This script needs river_client, a private River SDK that is not public. Its logged outputs are in runs/.")
from judge import load_tasks, evaluate, task_prompt
from fw import extract_json

BASE, STEPS, BATCH, EVAL_N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
TAG = sys.argv[5] if len(sys.argv) > 5 else "run"
SYS = "You are an expert control engineer tuning PID loops for robot joint actuators."
LR = float(os.environ.get("LR", "2e-4")); RANK = int(os.environ.get("RANK", "32"))
rend = get_renderer(BASE, thinking=False)
tok = rend.tokenizer
EOS = tok.eos_token_id if hasattr(tok, "eos_token_id") else None

def datum(msgs):
    ex = rend.build_training_example(msgs)
    ids, w = list(ex.input_ids), list(ex.weights)
    return {"input_ids": ids, "target_tokens": ids[1:] + [ids[-1]], "weights": w[1:] + [0.0]}

rows = [json.loads(l) for l in open(os.environ.get("TRAIN", "data/pid/train.jsonl"))]
UP = int(os.environ.get("UPSAMPLE_SO_UNSTABLE", "1"))
rows = rows + [r for r in rows if r["task"]["category"].startswith("so_unstable") for _ in range(UP - 1)]
random.Random(0).shuffle(rows)
data = [datum(r["messages"]) for r in rows]
print(f"{len(data)} train examples; mean len {sum(len(d['input_ids']) for d in data)/len(data):.0f} tok; loss tok {sum(sum(d['weights']) for d in data)/len(data):.1f}", flush=True)

by = defaultdict(list)
for t in load_tasks(): by[t["category"]].append(t)
evalset = [t for c in sorted(by) for t in by[c][:EVAL_N]]
stops = rend.get_stop_strings() if hasattr(rend, "get_stop_strings") else None

def prompt_for(t):
    return rend.build_text_prompt([{"role": "system", "content": SYS}, {"role": "user", "content": task_prompt(t)}])

def run_eval(sample_fn, label):
    t0 = time.time()
    def one(t):
        for a in range(5):
            try:
                txt = sample_fn(prompt_for(t)); break
            except Exception as e:
                txt = ""; time.sleep(3 * (a + 1))
        j = extract_json(txt) or {}
        ev = evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd")) if j else {"success": False}
        return {"uid": t["uid"], "category": t["category"], "gains": j, "raw": txt[:200], "success": bool(ev["success"]), "score": float(ev.get("score", 0))}
    with cf.ThreadPoolExecutor(8) as ex: res = list(ex.map(one, evalset))
    agg = defaultdict(list)
    for r in res: agg[r["category"]].append(r["success"])
    s = sum(r["success"] for r in res)
    print(f"[{label}] ControlEval success {s}/{len(res)} = {s/len(res):.1%} partial {sum(r['score'] for r in res)/len(res):.2f} ({time.time()-t0:.0f}s) | " +
          " ".join(f"{c.replace('first_order','fo').replace('second_order','so')}={sum(v)}/{len(v)}" for c, v in sorted(agg.items())), flush=True)
    return res

client = river.Client(api_key=os.environ["RIVER_API_KEY"])
log = {"base": BASE, "steps": STEPS, "batch": BATCH, "lr": LR, "rank": RANK, "evals": []}
with client.session(project="pid-tuner", experiment=TAG) as session:
    model = session.create_model(base_model=BASE, tokenizer=tok, lora=river.LoraConfig(rank=RANK, seed=0))
    print("model", model.model_id, flush=True)
    samp = lambda p: model.sample(p, max_tokens=60, temperature=0.0, stop=stops)[0][0].text
    if os.environ.get("EVAL_BEFORE", "1") == "1":
        r0 = run_eval(samp, "step 0 (base)"); log["evals"].append({"step": 0, "success": sum(x["success"] for x in r0)})
    i = 0
    for step in range(1, STEPS + 1):
        batch = data[i:i + BATCH]; i += BATCH
        if i >= len(data): i = 0
        fb = model.forward_backward(batch, loss_fn="cross_entropy")
        model.optim_step(lr=LR, grad_clip_norm=1.0)
        if step % 10 == 0 or step == 1: print(f"step {step} loss {fb.metrics.get('loss'):.4f}", flush=True)
    r1 = run_eval(samp, f"step {STEPS} (tuned)")
    log["evals"].append({"step": STEPS, "success": sum(x["success"] for x in r1)})
    ck = model.save_weights(f"pid-{TAG}", mode="inference")
    log["checkpoint"] = ck.path; print("checkpoint", ck.path, flush=True)
    json.dump({"log": log, "final": r1}, open(f"runs/river_sft_{TAG}.json", "w"), indent=1)
