#!/bin/bash
# TF32 for the fp32 action heads (tree starVLA_tf32): upstream, all, all + tf32, each in this job; then the loss of
# the 220 timed steps is compared by coverage/collect_others.py.
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_tf32
R=runs/others_tf32
Q3=$A/models/Qwen3-VL-4B-Instruct
Q25=$A/models/Qwen2.5-VL-3B-Instruct
cd $WS && mkdir -p $R/cov
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1)"
cov() {  # cov <name> <tag> <tree> <fast> <model> <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    say "$name / $tag"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model --set "$@" 2>&1 | tail -n 1 | cut -c1-400
}
combo() {  # combo <name> <model> <overrides...>
    local name=$1 model=$2; shift 2
    cov $name ref targets/reference "" $model "$@"
    cov $name all $T all $model "$@"
    cov $name all_tf32 $T all,tf32 $model "$@"
}
combo pi_q3 $Q3 framework.name=QwenPI
combo gr00t_q3 $Q3 framework.name=QwenGR00T
say "done"
