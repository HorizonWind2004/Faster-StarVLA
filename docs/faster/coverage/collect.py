"""Collect the validation results of the unified starVLA.fast interface into one file (runs/unified/coverage.json).

    python3 coverage/collect.py [--md]

Timings are only compared within one session (one batch job): the baseline is host-bound, and its step time was
905 ms in the session in which the reference timing was frozen and 796 - 816 ms in the two later ones (the node of the
first session was rebooted in between; the other node gives the lower figure too). SESSIONS says where each job ran.
"""
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
R = WS / "runs/unified"
SESSIONS = {
    "72220": {"session": 1, "node": "A", "when": "22-23 Nov", "note": "the reference timing was frozen in this session"},
    "72551": {"session": 2, "node": "B", "when": "23 Nov, afternoon", "note": "a second node of the same hardware model"},
    "72622": {"session": 3, "node": "A", "when": "23 Nov, evening", "note": "node A after it was rebooted on 23 Nov"},
}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def timing(combo, names):
    for name in names:
        d = load(R / "cov" / combo / name)
        if d and d.get("status") == "OK":
            return {"step_ms": d["step_ms"], "window_step_ms": d["window_step_ms"], "peak_mem_gib": d["peak_mem_gib"],
                    "gpu_util": d["gpu_util_mean_second_half"], "tree_digest": d.get("tree_digest")}
    return None


def judge(name, path, job, what):
    d = load(path)
    if not d:
        return None
    gates = d.get("gates", {})
    benefit = gates.get("benefit") or {}
    perf = d.get("performance") or {}
    refs = benefit.get("reference_step_ms")
    row = {"name": name, "what": what, "job": job, "session": SESSIONS[job]["session"], "node": SESSIONS[job]["node"], "mode": d.get("mode"), "status": d["status"],
           "tier": d.get("tier"), "switches": d.get("switches"), "tree_digest": d.get("tree_digest"), "started": d.get("started"),
           "reference_step_ms": perf.get("reference_step_ms"), "reference_runs_ms": refs if isinstance(refs, list) else None,
           "reference_is_frozen": not isinstance(refs, list), "candidate_step_ms": perf.get("candidate_step_ms"),
           "speedup": perf.get("speedup"), "invalid": benefit.get("invalid") or None}
    bitwise = gates.get("bitwise")
    if isinstance(bitwise, dict):
        row["tensors"], row["tensors_identical"] = bitwise.get("checked"), bitwise.get("bitwise_identical")
    statistical = gates.get("statistical")
    if isinstance(statistical, dict) and "endpoints" in statistical:
        row["max_abs_sigma"] = max(abs(e["mean_in_sigma"]) for e in statistical["endpoints"].values())
    return row


judges = [j for j in (
    judge("first release, exact set", WS / "runs/check_bitwise/check.json", "72220", "per-path variables, before the refactor"),
    judge("first release, all", WS / "runs/final/check.json", "72220", "per-path variables, before the refactor"),
    judge("unified, exact", R / "check_exact2/check.json", "72551", "STARVLA_FAST=exact, commit 7bf6a48"),
    judge("unified, all", R / "check_all/check.json", "72551", "STARVLA_FAST=all, commit 7bf6a48"),
    judge("unified final tree, exact", R / "final_exact/check.json", "72622", "STARVLA_FAST=exact, final commit"),
    judge("unified final tree, all", R / "final_all/check.json", "72622", "STARVLA_FAST=all, final commit"),
) if j]
by_name = {j["name"]: j for j in judges}

# ---- the headline recipe in each session: baseline and candidates measured in the same job
def headline(session, ref_row, exact_row, all_row):
    """ref_row: the three-process check whose two reference runs give this session's baseline."""
    if not (ref_row and ref_row["reference_runs_ms"]):
        return None
    ref = ref_row["reference_step_ms"]
    out = {"session": session, "node": ref_row["node"], "baseline_ms": ref, "baseline_runs_ms": ref_row["reference_runs_ms"],
           "all_ms": all_row["candidate_step_ms"] if all_row else None, "exact_ms": exact_row["candidate_step_ms"] if exact_row else None}
    out["all_speedup"] = ref / out["all_ms"] if out["all_ms"] else None
    out["exact_speedup"] = ref / out["exact_ms"] if out["exact_ms"] else None
    if session == 1 and exact_row:  # the frozen timing is from this session: the judge's own figure
        out["exact_speedup"] = exact_row["speedup"]
    out["final_check"] = ref_row["status"] + (f" {ref_row['tier']}" if ref_row.get("tier") else "")
    return out


g = by_name.get
sessions = [h for h in (headline(1, g("first release, all"), g("first release, exact set"), g("first release, all")),
                        headline(2, g("unified, all"), g("unified, exact"), g("unified, all")),
                        headline(3, g("unified final tree, all"), g("unified final tree, exact"), g("unified final tree, all"))) if h]

# ---- step times in session 2 (job 72551): every row has its own baseline from the same job
rows = []
ue, ua = by_name.get("unified, exact"), by_name.get("unified, all")
if ua and ua["reference_runs_ms"]:
    rows.append({"framework": "QwenOFT", "backbone": "Qwen3-VL-4B", "attention": "sdpa", "source": "judge processes (check_all, check_exact2)",
                 "ref": {"step_ms": ua["reference_step_ms"], "runs_ms": ua["reference_runs_ms"]},
                 "exact": {"step_ms": ue["candidate_step_ms"]} if ue else None, "all": {"step_ms": ua["candidate_step_ms"]}})
COMBOS = [("gr00t_q3", "QwenGR00T", "Qwen3-VL-4B", "sdpa"), ("pi_q3", "QwenPI", "Qwen3-VL-4B", "sdpa"),
          ("oft_q25_sdpa", "QwenOFT", "Qwen2.5-VL-3B", "sdpa"), ("oft_q25_fa2", "QwenOFT", "Qwen2.5-VL-3B", "flash_attention_2"),
          ("gr00t_q25_fa2", "QwenGR00T", "Qwen2.5-VL-3B", "flash_attention_2")]
reruns = {}
for combo, framework, backbone, attention in COMBOS:
    row = {"framework": framework, "backbone": backbone, "attention": attention, "source": f"cov/{combo}",
           **{tag: timing(combo, (f"{tag}_summary.json", f"{tag}.result.json")) for tag in ("ref", "exact", "all")}}
    if attention == "flash_attention_2" and row["all"] is None:
        row["all_equals_exact"] = True  # no further fast path applies under flash-attention
    rows.append(row)
    rerun = timing(combo, ("all_fixed_summary.json",))  # job 72622 (session 3), final tree: not comparable with the rows above
    if rerun or (R / "cov" / combo / "all_fixed.log").exists():
        reruns[combo] = {"framework": framework, "backbone": backbone, "attention": attention, "session": 3, "job": "72622",
                         "ran": rerun is not None, **(rerun or {})}

bitwise = []
suite = R / "bitwise4" if list((R / "bitwise4").glob("*.json")) else R / "bitwise3"
for p in sorted(suite.glob("*.json")):
    d = load(p)
    bitwise.append({"check": p.stem, "framework": d.get("framework"),
                    "backbone": d["backbone"] if "Qwen" in d["backbone"] else "Qwen3-VL-4B-Instruct",
                    "attention": d.get("attention", "sdpa"), "dtype": d.get("dtype", "bf16"), "applied": d["applied"],
                    "not_applicable": d["not_applicable"], "probes": d.get("vision_attn_probes"), "gradients": d["gradients"],
                    "gradients_different": d["gradients_different"], "loss_identical": d.get("loss_identical"),
                    "actions_identical": d.get("actions_identical"), "hidden_identical": d.get("hidden_identical"),
                    "reference_reproducible": d["reference_reproducible"], "PASS": d["PASS"]})
out = {"sessions": {v["session"]: {k: v[k] for k in ("node", "when", "note")} | {"job": job} for job, v in SESSIONS.items()},
       "headline_by_session": sessions, "judge": judges, "timing_session_2": rows, "reruns_final_tree_session_3": reruns, "bitwise_suite": suite.name, "bitwise": bitwise,
       "libero": load(WS.parent.parent / "libero/runs/e2e_unified/summary_e2e.json"),
       "libero_before": {k: v for k, v in (load(WS.parent.parent / "libero/runs/e2e_72408/summary_e2e.json") or {}).items()
                         if k == "M_pin_queue_c14"}}
(R / "coverage.json").write_text(json.dumps(out, indent=1) + "\n")

if "--md" in sys.argv:
    def cell(row, tag):
        t = row.get(tag)
        if not t:
            return "= exact" if tag == "all" and row.get("all_equals_exact") else "—"
        ref = (row.get("ref") or {}).get("step_ms")
        return f"{t['step_ms']:.1f}" + (f" ({ref / t['step_ms']:.2f}×)" if tag != "ref" and ref else "")
    print("| Framework | Backbone | Attention | baseline | exact | all |\n|---|---|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['framework']} | {r['backbone']} | {r['attention']} | {cell(r, 'ref')} | {cell(r, 'exact')} | {cell(r, 'all')} |")
    print()
    for h in sessions:
        print(f"session {h['session']} (node {h['node']}): baseline {h['baseline_ms']:.1f}  exact {h['exact_ms']:.1f} ({h['exact_speedup']:.2f}x)  "
              f"all {h['all_ms']:.1f} ({h['all_speedup']:.2f}x)  final check {h['final_check']}")
    print()
    for j in judges:
        print(f"| {j['name']} | session {j['session']} | {j['status']} {j.get('tier') or ''} | ref {j['reference_step_ms'] and round(j['reference_step_ms'], 1)}"
              f"{' (frozen)' if j['reference_is_frozen'] else ''} | cand {j['candidate_step_ms'] and round(j['candidate_step_ms'], 1)} | "
              f"{j['speedup'] and round(j['speedup'], 3)} | {j.get('tensors_identical')}/{j.get('tensors')} | {j.get('invalid')} |")
    print()
    for k, v in reruns.items():
        print("rerun", k, v.get("ran"), v.get("step_ms"))
    print()
    for b in bitwise:
        print(f"| {b['check']} | {b['framework'] or 'backbone'} | {b['backbone']} | {b['attention']} | {b['dtype']} | "
              f"{', '.join(b['applied'])} | n/a: {', '.join(b['not_applicable']) or '-'} | {b['gradients']} | {'PASS' if b['PASS'] else 'FAIL'} |")
