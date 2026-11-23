#!/bin/bash
# Validation of the unified starVLA.fast interface, run inside one 2-GPU allocation:
#   1. real-weights bitwise check of the backbone fast paths (Qwen3-VL-4B, Qwen2.5-VL-3B), one GPU each
#   2. the frozen judge on the refactored tree: exact set (gates 0-2), then the final check with everything on
#   3. step time of other framework / backbone combinations: upstream, STARVLA_FAST=exact, STARVLA_FAST=all
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
R=runs/unified
cd $WS && mkdir -p $R/cov $R/bitwise
say() { echo "=== $(date +%H:%M:%S) $*"; }

say "1. bitwise on real weights"
( source $A/env.sh
  CUDA_VISIBLE_DEVICES=0 python coverage/gpu_bitwise.py $WS/starVLA $A/models/Qwen3-VL-4B-Instruct $R/bitwise/qwen3.json > $R/bitwise/qwen3.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python coverage/gpu_bitwise.py $WS/starVLA $A/models/Qwen2.5-VL-3B-Instruct $R/bitwise/qwen2_5.json > $R/bitwise/qwen2_5.log 2>&1 &
  wait )
tail -n 1 $R/bitwise/qwen3.log $R/bitwise/qwen2_5.log | cut -c1-600

say "2a. judge, STARVLA_FAST=exact"
python3 bench/bench.py check --tree starVLA --config switches/unified_exact.json --out $R/check_exact 2>&1 | tail -n 3

cov() {  # cov <name> <tag> <tree> <fast> <model> <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    say "3. $name / $tag"
    python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model --set "$@" 2>&1 | tail -n 1
}
Q3=$A/models/Qwen3-VL-4B-Instruct
Q25=$A/models/Qwen2.5-VL-3B-Instruct
combo() {  # combo <name> <model> <overrides...>: upstream, exact, all
    local name=$1 model=$2; shift 2
    cov $name ref targets/reference "" $model "$@"
    cov $name exact starVLA exact $model "$@"
    cov $name all starVLA all $model "$@"
}
combo gr00t_q3 $Q3 framework.name=QwenGR00T
combo oft_q25_sdpa $Q25 framework.name=QwenOFT framework.qwenvl.attn_implementation=sdpa
combo oft_q25_fa2 $Q25 framework.name=QwenOFT
combo gr00t_q25_fa2 $Q25 framework.name=QwenGR00T
combo pi_q3 $Q3 framework.name=QwenPI

say "2b. judge, final check, STARVLA_FAST=all"
touch task/final-evaluation
python3 bench/bench.py check --final --tree starVLA --config switches/unified_all.json --out $R/check_all 2>&1 | tail -n 3
rm -f task/final-evaluation

say "2c. judge, variables of the first release (gates 0 and 2 only)"
python3 bench/bench.py check --skip-timing --tree starVLA --config switches/bitwise.json --out $R/check_legacy 2>&1 | tail -n 3
say "done"
