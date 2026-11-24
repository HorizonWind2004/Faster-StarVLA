#!/bin/bash
# The remaining Qwen frameworks, flash-attention configs with the sdpa path, and TF32 (tree starVLA_r13), in one
# 2-GPU allocation. Every step runs whether or not the one before it succeeded.
#   1. real-weights bitwise check per new framework (exact paths, training batch from data-loader workers), and the
#      distance between a flash-attention step and the same step after the sdpa path
#   2. the Qwen2.5-VL flash-attention configs again: `all` now switches them to SDPA
#   3. the headline recipe with TF32, and the frozen judge on STARVLA_FAST=all,tf32 (gates 0-3)
#   4. step time per new framework: upstream, exact, all, all + tf32
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_r13
R=runs/r13
C=$WS/coverage/configs
Q3=$A/models/Qwen3-VL-4B-Instruct
Q25=$A/models/Qwen2.5-VL-3B-Instruct
cd $WS && mkdir -p $R/bitwise $R/cov
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1)"

say "1. bitwise on real weights, training batch from data-loader workers; flash-attention against sdpa"
( source $A/env.sh
B=$WS/$R/bitwise
fw() { CUDA_VISIBLE_DEVICES=$1 python coverage/framework_bitwise.py $T $3 $4 sdpa $B/$2.json > $B/$2.log 2>&1; }
fw 0 fw_piv3_q3 QwenPI_v3 $Q3 & CHECK_CONFIG=$C/discrete_diffusion.yaml fw 1 fw_dd_q3 QwenDiscreteDiffusion $Q3 & wait
CHECK_CONFIG=$C/adapter.yaml CHECK_ACTION_STEPS=8 fw 0 fw_adapter_q3 QwenAdapter $Q3 & CHECK_CONFIG=$C/langforce.yaml fw 1 fw_langforce_q3 LangForce $Q3 & wait
CHECK_CONFIG=$C/qwendual.yaml fw 0 fw_dual_q3 QwenDual $Q3 &
CUDA_VISIBLE_DEVICES=1 python coverage/sdpa_switch.py $T QwenOFT $Q25 $B/sdpa_oft_q25.json > $B/sdpa_oft_q25.log 2>&1 & wait
CUDA_VISIBLE_DEVICES=0 python coverage/sdpa_switch.py $T QwenGR00T $Q25 $B/sdpa_gr00t_q25.json > $B/sdpa_gr00t_q25.log 2>&1
tail -q -n 1 $B/*.log | cut -c1-600 )

cov() {  # cov <name> <tag> <tree> <fast> <model> <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    say "$name / $tag"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model --set "$@" 2>&1 | tail -n 1 | cut -c1-400
}
runs() {  # runs <name> <model> "<tags>" <overrides...>: ref = upstream, the others = STARVLA_FAST levels
    local name=$1 model=$2 tags=$3; shift 3
    for tag in $tags; do
        case $tag in
            ref) cov $name ref targets/reference "" $model "$@" ;;
            all_tf32) cov $name all_tf32 $T all,tf32 $model "$@" ;;
            *) cov $name $tag $T $tag $model "$@" ;;
        esac
    done
}
say "2. flash-attention configs"
runs oft_q25_fa2 $Q25 "ref exact all" framework.name=QwenOFT
runs gr00t_q25_fa2 $Q25 "ref exact all all_tf32" framework.name=QwenGR00T

say "3. headline recipe with TF32, and the judge"
runs oft_q3 $Q3 "ref all all_tf32" framework.name=QwenOFT
python3 bench/bench.py check --tree starVLA_r13 --config switches/unified_all_tf32.json --out $R/judge_all_tf32 2>&1 | tail -n 4

say "4. step times of the other frameworks"
runs piv3_q3 $Q3 "ref exact all all_tf32" framework.name=QwenPI_v3
runs dd_q3 $Q3 "ref exact all all_tf32" config_yaml=$C/discrete_diffusion.yaml framework.name=QwenDiscreteDiffusion
runs adapter_q3 $Q3 "ref exact all" config_yaml=$C/adapter.yaml framework.name=QwenAdapter
runs langforce_q3 $Q3 "ref exact all all_tf32" config_yaml=$C/langforce.yaml framework.name=LangForce
runs dual_q3 $Q3 "ref exact all all_tf32" config_yaml=$C/qwendual.yaml framework.name=QwenDual
say "done"
