"""Collect the validation results of the unified starVLA.fast interface into one file (runs/unified/coverage.json).

    python3 coverage/collect.py [--md]

Timings are only compared within one session (one batch job): the baseline is host-bound, and its step time was
905 ms in the session in which the reference timing was frozen and 796 - 816 ms in the two later ones (the node of the
first session was rebooted in between; the other node gives the lower figure too). SESSIONS says where each job ran.
Sessions 4 and later are the runs after worker_prep was made to work for every framework and Qwen backbone;
`timing_by_session` has every combination measured in them, and `coverage` the latest measurement of each.
"""
import json
import itertools
import statistics
import sys
from pathlib import Path

WS = Path(__file__).resolve().parent.parent
R = WS / "runs/unified"
SESSIONS = {
    "72220": {"session": 1, "node": "A", "when": "22-23 Nov", "note": "the reference timing was frozen in this session"},
    "72551": {"session": 2, "node": "B", "when": "23 Nov, afternoon", "note": "a second node of the same hardware model"},
    "72622": {"session": 3, "node": "A", "when": "23 Nov, evening", "note": "node A after it was rebooted on 23 Nov"},
    "72688": {"session": 4, "node": "C", "when": "23 Nov, night", "note": "a third node; worker_prep for every framework"},
    "72693": {"session": 5, "node": "C", "when": "24 Nov", "note": "TF32 for the fp32 action heads"},
    "72750": {"session": 6, "node": "C", "when": "24 Nov", "note": "the other Qwen frameworks, sdpa path, judge with TF32"},
    "72753": {"session": 7, "node": "C", "when": "24 Nov", "note": "condition_once, unused_views, PI0 / PI05"},
    "72867": {"session": 9, "node": None, "when": "24 Nov", "note": "PI0 / PI05 with worker_examples and openpi_compile; all and max alternating"},
    "73035": {"session": 11, "node": None, "when": "24 Nov, evening", "note": "probes: optimizer, stages, new paths on and off"},
    "73045": {"session": 12, "node": None, "when": "24-25 Nov", "note": "final tree: judge, the whole table, statistical gate"},
    "72879": {"session": 10, "node": None, "when": "24 Nov", "note": "statistical gate on QwenPI; judge on the released commit"},
    "73275": {"session": 13, "node": None, "when": "25 Nov, morning", "note": "the new backbones: Qwen3.5, Wan (and the distances of Gemma 4, MiniCPM, Florence-2)"},
    "73276": {"session": 13, "node": None, "when": "25 Nov", "note": "session 13 continued: Gemma 4 (eager), MiniCPM-RobotManip, Florence-2, Qwen3.5 + GR00T"},
    "73509": {"session": 13, "node": None, "when": "25 Nov, evening", "note": "session 13 continued: Gemma 4 at batch 4, Qwen3.5 (ref / exact)"},
    "73564": {"session": 13, "node": None, "when": "25 Nov, evening", "note": "Qwen3.5 all / max: compile_lm leaves linear-attention models alone"},
    "73603": {"session": 13, "node": None, "when": "25 Nov, night", "note": "Gemma 4 all / max: the causal mask is built for unpadded batches too"},
    "72818": {"session": 8, "node": "C", "when": "24 Nov", "note": "final tree: judge, worker_examples and openpi_sdpa for PI0 / PI05"},
}
OTHERS = WS / "runs/others"  # session 4
TF32 = WS / "runs/others_tf32"  # session 5
ROOTS = {4: OTHERS, 5: TF32, 6: WS / "runs/r13", 7: WS / "runs/r14", 8: WS / "runs/r15", 9: WS / "runs/r16", 10: WS / "runs/r17",
         11: WS / "runs/r20", 12: WS / "runs/r21", 13: WS / "runs/r22"}
TAGS = ("ref", "exact", "all", "all_tf32")
Q3, Q25, SDPA, FA2 = "Qwen3-VL-4B", "Qwen2.5-VL-3B", "sdpa", "flash_attention_2"
COMBO_INFO = {  # run directory -> framework, backbone, attention the config asks for
    "oft_q3": ("QwenOFT", Q3, SDPA), "gr00t_q3": ("QwenGR00T", Q3, SDPA), "pi_q3": ("QwenPI", Q3, SDPA),
    "piv3_q3": ("QwenPI_v3", Q3, SDPA), "dd_q3": ("QwenDiscreteDiffusion", Q3, SDPA),
    "adapter_q3": ("QwenAdapter", Q3, SDPA), "langforce_q3": ("LangForce", Q3, SDPA), "dual_q3": ("QwenDual", Q3, SDPA),
    "oft_q25_sdpa": ("QwenOFT", Q25, SDPA), "oft_q25_fa2": ("QwenOFT", Q25, FA2), "gr00t_q25_fa2": ("QwenGR00T", Q25, FA2),
    "pi0": ("PI0", "PaliGemma (openpi)", "eager"), "pi05": ("PI05", "PaliGemma (openpi)", "eager"),
    # session 13, transformers 5 for Qwen3.5 and Gemma 4 (the attention each recipe ends up with)
    "q35_gr00t": ("QwenGR00T", "Qwen3.5-0.8B", FA2), "q35_oft": ("QwenOFT", "Qwen3.5-0.8B", FA2),
    "gemma4_pi_b": ("Gemma4PI", "Gemma-4-E2B (batch 4)", "eager"), "wan_gr00t": ("WanGR00T", "Wan2.2-TI2V-5B (world model)", SDPA),
}


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def timing(combo, names, root=None):
    for name in names:
        d = load((root or R) / "cov" / combo / name)
        if d and d.get("status") == "OK":
            losses = d.get("losses_rank0") or []
            return {"step_ms": d["step_ms"], "window_step_ms": d["window_step_ms"], "peak_mem_gib": d["peak_mem_gib"],
                    "gpu_util": d["gpu_util_mean_second_half"], "tree_digest": d.get("tree_digest"),
                    "mean_loss_steps_100_220": sum(losses[100:220]) / len(losses[100:220]) if len(losses) >= 220 else None,
                    "loss_standard_error_steps_100_220": statistics.stdev(losses[100:220]) / 120 ** 0.5 if len(losses) >= 220 else None}
        if d and root is not None:  # a run of the later sessions that failed: keep why
            return {"step_ms": None, "status": d.get("status"), "error": str(d.get("error"))[:300]}
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
    judge("worker_prep for every framework, exact", OTHERS / "judge_exact/check.json", "72688", "STARVLA_FAST=exact, commit b14fd6b"),
    judge("all + tf32", ROOTS[6] / "judge_all_tf32/check.json", "72750", "STARVLA_FAST=all,tf32, commit 8332201"),
    judge("final tree, exact", ROOTS[8] / "judge_exact/check.json", "72818", "STARVLA_FAST=exact, commit e5f7e15"),
    judge("final tree, all", ROOTS[8] / "final_all/check.json", "72818", "STARVLA_FAST=all, commit e5f7e15, three-process check"),
    judge("commit 8427140, all", WS / "runs/r17/judge_all/check.json", "72879", "STARVLA_FAST=all, against the frozen timing"),
    judge("commit 8427140, exact", WS / "runs/r17/judge_exact/check.json", "72879", "STARVLA_FAST=exact"),
    judge("released commit, all", ROOTS[12] / "judge_all/check.json", "73045", "STARVLA_FAST=all (adamw_compiled, compile_head), against the frozen timing"),
    judge("released commit, exact", ROOTS[12] / "judge_exact/check.json", "73045", "STARVLA_FAST=exact"),
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
                        headline(3, g("unified final tree, all"), g("unified final tree, exact"), g("unified final tree, all")),
                        headline(8, g("final tree, all"), g("final tree, exact"), g("final tree, all"))) if h]

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

# ---- sessions 4 and later: every combination measured in a job, each with the baseline of that job
def session_rows(session):
    out = []
    for combo, (framework, backbone, attention) in COMBO_INFO.items():
        cells = {tag: timing(combo, (f"{tag}_summary.json",), ROOTS[session]) for tag in TAGS}
        if not cells["all_tf32"]:  # the precision level: `max`, which is `all,tf32` on the Qwen frameworks
            cells["all_tf32"] = timing(combo, ("max_summary.json",), ROOTS[session])
        if any(cells.values()):
            out.append({"framework": framework, "backbone": backbone, "attention": attention, "combo": combo,
                        "session": session, **cells})
    return out


by_session = {session: session_rows(session) for session in ROOTS}
latest = {}  # per combination, the row of the latest session that measured a baseline and a candidate
for session in sorted(ROOTS):
    for row in by_session[session]:
        if session == 8 and row["combo"] in ("pi0", "pi05"):
            continue  # `all` of that tree had openpi_sdpa (now level max) and no openpi_compile, and worker_examples did not apply
        if (row["ref"] or {}).get("step_ms") and any((row[tag] or {}).get("step_ms") for tag in TAGS[1:]):
            latest[row["combo"]] = row
coverage = [latest[combo] for combo in COMBO_INFO if combo in latest]
rows4, rows5 = by_session[4], by_session[5]
# session 9: PI0 / PI05, the levels all and max alternating in one job
pi_alternating = {combo: {tag: cell for tag in ("ref", "exact", "all", "max", "all2", "max2")
                          if (cell := timing(combo, (f"{tag}_summary.json",), ROOTS[9]))} for combo in ("pi0", "pi05")}


def permutation_p(a, b):
    """Two-sided exact permutation test on the difference of the means of two small samples."""
    pooled, observed, hits, total = a + b, abs(statistics.fmean(a) - statistics.fmean(b)), 0, 0
    for chosen in itertools.combinations(range(len(pooled)), len(a)):
        x = [pooled[i] for i in chosen]
        y = [pooled[i] for i in range(len(pooled)) if i not in chosen]
        total += 1
        hits += abs(statistics.fmean(x) - statistics.fmean(y)) >= observed * (1 - 1e-12)
    return hits / total


def stat_summary(folder):
    """The judge's paired verdicts as recorded by stat_gate.py, plus an unpaired comparison of each candidate's
    seeds with the reference's seeds (for a framework whose upstream does not reproduce itself on a seed)."""
    summary, frozen = load(folder / "summary.json"), load(folder / "frozen.json")
    if not summary or not frozen:
        return None
    reference = {name: [values[name] for _, values in sorted(frozen["reference_values"].items(), key=lambda kv: int(kv[0]))]
                 for name in frozen["endpoints"]}
    out = {"framework": summary["framework"], "reference_seeds": summary["seeds"], "sigma_k": summary["sigma_k"],
           "reference": {name: {"values": reference[name], "mean": statistics.fmean(reference[name]),
                                "sigma_seed": frozen["endpoints"][name]["sigma_seed"]} for name in reference},
           "candidates": {}}
    for fast, entry in summary["candidates"].items():
        if not entry.get("endpoints"):
            out["candidates"][fast] = entry
            continue
        seeds, endpoints = entry["seeds"], {}
        for name, e in entry["endpoints"].items():
            values = [frozen["reference_values"][str(seed)][name] + d for seed, d in zip(seeds, e["differences"])]
            sigma = frozen["endpoints"][name]["sigma_seed"]
            endpoints[name] = {"values": values, "mean": statistics.fmean(values),
                               "paired_mean_difference_in_sigma": e["difference_in_sigma"], "paired_state": e["state"],
                               "paired_differences_in_sigma": [d / sigma for d in e["differences"]] if sigma else None,
                               "difference_of_means_in_sigma": (statistics.fmean(values) - statistics.fmean(reference[name])) / sigma if sigma else None,
                               "permutation_p": permutation_p(values, reference[name]) if len(values) >= 3 else None}
        out["candidates"][fast] = {"seeds": seeds, "paired_verdict": entry["verdict"], "endpoints": endpoints}
    return out


stat_gate = {p.name: stat_summary(p) for p in sorted((WS / "runs/r17").glob("stat_*")) if (p / "summary.json").exists()}
# session 12 reran all / max on the final tree (all now with adamw_compiled and compile_head) against the same upstream seeds
for name, summary in stat_gate.items():
    final = ROOTS[12] / name
    if summary and (final / "summary.json").exists():
        newer = stat_summary(final)
        for fast, entry in newer["candidates"].items():
            summary["candidates"][fast] = {**entry, "session": 12}
        summary["candidates_from_session"] = {fast: 12 if fast in newer["candidates"] else 10 for fast in summary["candidates"]}
# session 8: the headline recipe on the final tree and on the tree of session 4, alternating in one job
alternating = {tag: timing("oft_q3", (f"{tag}_summary.json",), ROOTS[8])
               for tag in ("ref", "exact", "all", "all_old", "all2", "all_old2", "all_tf32", "ref2")}
alternating = {tag: cell for tag, cell in alternating.items() if cell}

# ---- rounding-level paths: distance of one training step from the default step
distances = []
for session, folder in ((6, "bitwise"), (7, "distance"), (8, "distance"), (10, "distance"), (11, "distance"), (12, "distance"), (13, "distance")):
    for p in sorted((ROOTS[session] / folder).glob("*.json")):
        d = load(p)
        if d and ("default_vs_fast" in d or "flash_vs_sdpa" in d):
            fast, noise = d.get("default_vs_fast") or d["flash_vs_sdpa"], d.get("default_vs_default") or d["flash_vs_flash"]
            distances.append({"check": p.stem, "session": session, "framework": d["framework"], "backbone": d["backbone"],
                              "loaded_with": d["loaded_with"], "paths": d.get("paths", "sdpa"), "applied": d["applied"],
                              "noise_floor_gradient_relative_difference": noise["gradient_relative_difference"],
                              **{k: fast[k] for k in ("loss", "loss_relative_difference", "gradient_relative_difference",
                                                      "gradient_cosine")}, "by_module": fast.get("by_module"),
                              "loss_kind": d.get("loss") if isinstance(d.get("loss"), str) else "action"})

bitwise = []
suite = R / "bitwise4" if list((R / "bitwise4").glob("*.json")) else R / "bitwise3"
newer = {p.stem: (4, p) for p in sorted((OTHERS / "bitwise").glob("*.json"))}  # the framework checks with worker_prep
newer.update({p.stem: (6, p) for p in sorted((ROOTS[6] / "bitwise").glob("fw_*.json"))})
for stem, (session, p) in sorted({**{q.stem: (3, q) for q in suite.glob("*.json")}, **newer}.items()):
    d = load(p)
    workers = d.get("worker_prep") or {}
    # a check passes when everything is identical and the training batch's inputs all came from the workers; a
    # gradient that differs between two runs of the default code itself cannot be compared and is listed apart
    unstable = d.get("reference_rerun_gradients_different", 0)
    identical = bool(d.get("loss_identical", True) and d.get("actions_identical", True) and d.get("hidden_identical", True)
                     and d["gradients_different"] <= unstable and (not workers or (workers["used"] >= 1 and not workers["other prompt"])))
    bitwise.append({"check": p.stem, "framework": d.get("framework"), "session": session,
                    "worker_prep_batches": workers.get("used"), "gradients_not_reproducible_upstream": unstable,
                    "first_different": d.get("first_different"), "identical": identical,
                    "backbone": d["backbone"] if "Qwen" in d["backbone"] else "Qwen3-VL-4B-Instruct",
                    "attention": d.get("attention", "sdpa"), "dtype": d.get("dtype", "bf16"), "applied": d["applied"],
                    "not_applicable": d["not_applicable"], "probes": d.get("vision_attn_probes"), "gradients": d["gradients"],
                    "gradients_different": d["gradients_different"], "loss_identical": d.get("loss_identical"),
                    "actions_identical": d.get("actions_identical"), "hidden_identical": d.get("hidden_identical"),
                    "reference_reproducible": d["reference_reproducible"], "PASS": d["PASS"]})
out = {"sessions": {v["session"]: {k: v[k] for k in ("node", "when", "note")} | {"job": job} for job, v in SESSIONS.items()},
       "headline_by_session": sessions, "judge": judges, "coverage": coverage,
       "timing_by_session": {str(k): v for k, v in by_session.items()}, "rounding_distance": distances,
       "headline_alternating_session_8": alternating, "openpi_alternating_session_9": pi_alternating,
       "statistical_gate_other_frameworks": stat_gate,
       "timing_session_4": rows4, "timing_session_5_tf32": rows5, "timing_session_2": rows,
       "reruns_final_tree_session_3": reruns, "bitwise_suite": suite.name, "bitwise": bitwise,
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
    for session, session_rows_ in by_session.items():
        print(f"\nsession {session}")
        for r in session_rows_:
            cells = " | ".join((cell(r, tag) if (r.get(tag) or {}).get("step_ms") else ("ERR" if r.get(tag) else "—")) for tag in TAGS)
            print(f"| {r['framework']} | {r['backbone']} | {r['attention']} | {cells} |")
    print("\nrounding distance")
    for d in distances:
        modules = "; ".join(f"{k}: {v['relative_difference']:.3g} (cos {v['cosine']:.4f})" for k, v in (d["by_module"] or {}).items())
        print(f"| {d['check']} | s{d['session']} | {d['paths']} | loss {d['loss_relative_difference']:.2e} | grad {d['gradient_relative_difference']:.3g} "
              f"cos {d['gradient_cosine']:.4f} | noise {d['noise_floor_gradient_relative_difference']:.1e} | {modules} |")
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
              f"{', '.join(b['applied'])} | n/a: {', '.join(b['not_applicable']) or '-'} | {b['gradients']} | "
              f"{'identical' if b['identical'] else 'DIFFERENT'}"
              f"{' (%d gradient(s) not reproducible upstream: %s)' % (b['gradients_not_reproducible_upstream'], ', '.join(b['first_different'] or [])) if b['gradients_not_reproducible_upstream'] else ''} |")
