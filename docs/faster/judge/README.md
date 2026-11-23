# The frozen judge of this task (archived copy)

- `GOAL.md`: the task statement the optimizing agent worked from.
- `bench/`: `bench.py` (commands and gates), `timing.py` (step timing of the real entry point), `starvla_adapter.py`
  (drives the tree's own training loop for the bitwise records and trajectories), `loader_probe.py`, the cases and the
  trajectory endpoints.
- `numerics/`: the generic recorder and comparison code of
  [ResearchBoost-Agent](https://github.com/DAGroup-PKU/ResearchBoost-Agent/tree/main/numerics), copied unchanged.
- `task/`: what was frozen before optimization: `timing_ref.json` (reference timing), `records/` (manifests of the
  bitwise reference records: names, shapes, norms and SHA-256 of every recorded tensor; the tensor sketches themselves,
  430 MB, are not included), `traj/` (reference trajectories of 6 seeds), `traj_frozen.json` (endpoints and sigma),
  `loader_reference.json`, `trainable.json`, `formulation.json` (evaluator hash, acceptance results),
  `judge_changes.md` (the one change made after freezing, with its regression record), `baseline.md`.

Host-specific absolute paths in these files were replaced by `$WORK`, `$ENVS`, `$HOME_DIR`; the evaluator hash in
`formulation.json` refers to the files as they were on the measurement host.

To rerun the judge elsewhere: lay out a workspace as in `GOAL.md` (`targets/reference` = starVLA `99764f5`, the
candidate tree, `bench/`, `numerics/`, `task/`), point `bench/bench.json` (`python`, `env_file`, `gpu_lock`, `assets`)
at your environment, and re-record the reference (`bench.py baseline --freeze`, `bench.py formulate records`,
`bench.py formulate traj`): timings and bitwise records are specific to the GPU model and library versions.
