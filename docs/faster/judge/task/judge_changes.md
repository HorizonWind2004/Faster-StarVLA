# Judge changes

## Change 1 (2026-11-22): how the adapter finds the dataset for the probe batch

**Kind**: implementation fix in `bench/starvla_adapter.py`; no gate, threshold, endpoint or frozen record changed.

**Old**: `_make_probe` located the tree's dataset by scanning `gc.get_objects()` for a `DataLoader` whose dataset has
`sample_step`.

**New**: the adapter wraps `torch.utils.data.DataLoader.__init__` and remembers the dataset of the first loader built on
a dataset with `sample_step`; `_make_probe` uses it. Same dataset object, same indices
(`bench.json` `trajectory.probe_index`, `probe_rows`), same forward.

**Trigger**: the first `check --final` of the candidate (switch set `switches/final.json`) passed gates 0 and 1
(speedup 2.0010), was not bitwise at gate 2 as expected for the compile switches, and then ended ERROR at gate 3:
`RuntimeError: no starVLA mixture dataset found for the probe batch` (`runs/final_attempt1/`). The candidate calls
`gc.freeze()` after its compiled graphs exist; objects in the permanent generation are not returned by
`gc.get_objects()`, so the scan no longer saw the DataLoader. The judge could not evaluate the candidate; it did not
reject it. Nothing in the baseline or in any gate depends on the scan.

**Regression** (old rules vs new rules):

| Run | Old adapter | New adapter |
|---|---|---|
| reference, trajectory seed 0 | frozen record `task/traj/ref/traj_s0.r*.json` | re-recorded: all 640 values (loss, probe, upd, upd_vlm, updq, upd_vlmq on both ranks, 60 steps) identical to the frozen record |
| control arm, every learning rate x 1.02, seed 0 | INCORRECT (updq +16.38 sigma, updvlmq +14.72 sigma) | re-run after the change, see `task/judge_changes_control.result (same verdict: INCORRECT, updq and updvlmq beyond the bound)` |
| bitwise records (gate 2) | the probe is not used by record runs | unchanged |

The frozen timing, records, trajectories and sigma are the ones frozen before optimization. Evaluator hash:
`9a297649b4f4fc66b6ea7900d1a83c63ba064079391f3ec816f5fd9c56bb17ce` -> `868f97722fd37ab3fb2b34a8347147bf4214028f5bf084e73425065a78b68a95`.

**Who decided**: the formulating agent, with the owner asleep; flagged in the report for the owner to confirm.
