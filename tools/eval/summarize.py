"""Success rates from the per-episode logs of run_eval.sh: summarize.py <out dir> [...]"""
import glob, json, sys
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
for out in sys.argv[1:]:
    res = {}
    for s in SUITES:
        rows = [json.loads(line) for f in glob.glob(f"{out}/{s}/w*/episodes.jsonl") for line in open(f)]
        res[s] = {"episodes": len(rows), "success": round(100.0 * sum(r["success"] for r in rows) / len(rows), 2) if rows else None}
    done = [v["success"] for v in res.values() if v["success"] is not None]
    res["avg"] = round(sum(done) / len(done), 2) if len(done) == len(SUITES) else None
    res["episodes"] = sum(v["episodes"] for v in res.values() if isinstance(v, dict))
    json.dump(res, open(f"{out}/summary.json", "w"), indent=1)
    print(out, json.dumps(res))
