"""Fireworks pipeline for buildathon day (REST; firectl blocks agent-run mutations).
Steps: export -> upload dataset -> SFT job -> poll -> deployment -> smoke test.
DRY RUN by default; pass --go to actually call mutating endpoints (costs money: SFT + dedicated deployment).
Usage:
  python deploy/fireworks_pipeline.py export data/pid/train_v2.jsonl deploy/pid_train.jsonl
  python deploy/fireworks_pipeline.py upload deploy/pid_train.jsonl pid-train-v2 [--go]
  python deploy/fireworks_pipeline.py sft pid-train-v2 qwen3p5-9b pid-tuner-9b [--go]
  python deploy/fireworks_pipeline.py status sft <job_id>
  python deploy/fireworks_pipeline.py deploy pid-tuner-9b [--go]
  python deploy/fireworks_pipeline.py smoke <model_or_deployment_path>
"""
import os, sys, json, time, httpx
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
KEY = os.environ["FIREWORKS_API_KEY"]; ACCT = "accounts/sissiwang-xlorceau0c"; API = "https://api.fireworks.ai/v1"
H = {"Authorization": f"Bearer {KEY}"}; GO = "--go" in sys.argv; args = [a for a in sys.argv[1:] if a != "--go"]

def call(method, url, **kw):
    if method != "GET" and not GO:
        print(f"[dry-run] {method} {url}\n  {json.dumps(kw.get('json'), indent=1)[:800] if kw.get('json') else ''}"); return {}
    r = httpx.request(method, url, headers=H, timeout=600, **kw); print(method, url, r.status_code); r.raise_for_status(); return r.json() if r.text else {}

cmd = args[0]
if cmd == "export":  # keep only OpenAI-format messages
    src, dst = args[1], args[2]; n = 0
    with open(dst, "w") as f:
        for l in open(src):
            f.write(json.dumps({"messages": json.loads(l)["messages"]}) + "\n"); n += 1
    print(n, "rows ->", dst)
elif cmd == "upload":
    path, ds = args[1], args[2]; n = sum(1 for _ in open(path))
    call("POST", f"{API}/{ACCT}/datasets", json={"datasetId": ds, "dataset": {"userUploaded": {}, "exampleCount": str(n)}})
    if GO:
        with open(path, "rb") as fh: call("POST", f"{API}/{ACCT}/datasets/{ds}:upload", files={"file": (Path(path).name, fh)})
    else: print(f"[dry-run] upload {path} ({n} rows) to {ds}")
elif cmd == "sft":
    ds, base, out = args[1], args[2], args[3]
    body = {"dataset": f"{ACCT}/datasets/{ds}", "baseModel": f"accounts/fireworks/models/{base}", "outputModel": f"{ACCT}/models/{out}",
            "epochs": 2, "loraRank": 32, "learningRate": 0.0002, "batchSizeSamples": 32, "displayName": out}
    j = call("POST", f"{API}/{ACCT}/supervisedFineTuningJobs", json=body); print(json.dumps(j, indent=1)[:600])
elif cmd == "status":
    kind, jid = args[1], args[2]; path = {"sft": "supervisedFineTuningJobs", "dep": "deployments"}[kind]
    print(json.dumps(call("GET", f"{API}/{ACCT}/{path}/{jid}"), indent=1)[:1500])
elif cmd == "deploy":
    model = args[1]
    body = {"baseModel": f"{ACCT}/models/{model}", "displayName": model, "minReplicaCount": 1, "maxReplicaCount": 1}
    print(json.dumps(call("POST", f"{API}/{ACCT}/deployments", json=body), indent=1)[:800])
elif cmd == "smoke":
    from openai import OpenAI
    sys.path.insert(0, "control"); from judge import load_tasks, task_prompt, evaluate
    sys.path.insert(0, "."); from fw import extract_json
    c = OpenAI(api_key=KEY, base_url="https://api.fireworks.ai/inference/v1"); t = load_tasks()[0]
    r = c.chat.completions.create(model=args[1], messages=[{"role": "system", "content": "You are an expert control engineer tuning PID loops for robot joint actuators."},
                                                          {"role": "user", "content": task_prompt(t)}], max_tokens=80, temperature=0)
    txt = r.choices[0].message.content; j = extract_json(txt) or {}; print(txt, evaluate(t, j.get("Kp"), j.get("Ki"), j.get("Kd")).get("success"))
