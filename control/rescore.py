"""Re-evaluate saved gains in runs/pid_*.json with the current judge. Usage: python control/rescore.py runs/pid_X.json"""
import sys, json
from collections import defaultdict
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from judge import load_tasks, evaluate
T = {t["uid"]: t for t in load_tasks()}
for f in sys.argv[1:]:
    res = json.load(open(f)); agg = defaultdict(list); sc = []
    for r in res:
        g = r.get("gains") or {}
        ev = evaluate(T[r["uid"]], g.get("Kp"), g.get("Ki"), g.get("Kd")) if g else {"success": False}
        r["success"] = bool(ev["success"]); r["score"] = float(ev.get("score", 0)); agg[r["category"]].append(r["success"]); sc.append(r["score"])
    json.dump(res, open(f, "w"), indent=1)
    errs = sum(1 for r in res if r.get("error"))
    print(f"{f}: success {sum(r['success'] for r in res)}/{len(res)} (errors {errs}) partial {sum(sc)/len(sc):.2f} | " + " ".join(f"{c[:14]}={sum(v)}/{len(v)}" for c, v in sorted(agg.items())))
