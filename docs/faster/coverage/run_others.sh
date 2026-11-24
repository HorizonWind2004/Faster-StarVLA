#!/bin/bash
# worker_prep on every framework and Qwen backbone (tree starVLA_next), in one 2-GPU allocation:
#   1. real-weights check per framework x backbone: baseline against the exact paths, the candidate's training batch
#      coming out of a DataLoader with workers (so the model gets the inputs a worker built)
#   2. main-thread profile of QwenGR00T and QwenPI with STARVLA_FAST=exact (what is left on the host)
#   3. step time per combination: upstream, exact, all, each with its own baseline from this job
#   4. the frozen judge, exact set (gates 0-2): QwenOFT now takes the generic path as well
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_next
R=runs/others
Q3=$A/models/Qwen3-VL-4B-Instruct
Q25=$A/models/Qwen2.5-VL-3B-Instruct
cd $WS && mkdir -p $R/bitwise $R/cov
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1)"

say "1. bitwise on real weights, training batch from data-loader workers"
( source $A/env.sh
B=$WS/$R/bitwise
fw() { CUDA_VISIBLE_DEVICES=$1 python coverage/framework_bitwise.py $T $3 $4 $5 $B/$2.json > $B/$2.log 2>&1; }
fw 0 fw_oft_q3 QwenOFT $Q3 sdpa & fw 1 fw_gr00t_q3 QwenGR00T $Q3 sdpa & wait
fw 0 fw_pi_q3 QwenPI $Q3 sdpa & fw 1 fw_oft_q25 QwenOFT $Q25 sdpa & wait
fw 0 fw_gr00t_q25 QwenGR00T $Q25 sdpa & fw 1 fw_oft_q25_fa2 QwenOFT $Q25 flash_attention_2 & wait
fw 0 fw_gr00t_q25_fa2 QwenGR00T $Q25 flash_attention_2 & fw 1 fw_pi_q25_fa2 QwenPI $Q25 flash_attention_2 & wait
tail -q -n 1 $B/*.log | cut -c1-420 )

cov() {  # cov <name> <tag> <tree> <fast> <model> <extra time_cfg args> -- <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    say "$name / $tag"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model "${extra[@]}" --set "$@" 2>&1 | tail -n 1 | cut -c1-400
}
say "2. profiles"
cov gr00t_q3 prof_exact $T exact $Q3 --cprofile 20 -- framework.name=QwenGR00T
cov pi_q3 prof_exact $T exact $Q3 --cprofile 20 -- framework.name=QwenPI

say "3. step times"
combo() {  # combo <name> <model> <overrides...>: upstream, exact, all
    local name=$1 model=$2; shift 2
    cov $name ref targets/reference "" $model -- "$@"
    cov $name exact $T exact $model -- "$@"
    cov $name all $T all $model -- "$@"
}
combo gr00t_q3 $Q3 framework.name=QwenGR00T
combo pi_q3 $Q3 framework.name=QwenPI
combo oft_q25_fa2 $Q25 framework.name=QwenOFT
combo gr00t_q25_fa2 $Q25 framework.name=QwenGR00T
combo oft_q25_sdpa $Q25 framework.name=QwenOFT framework.qwenvl.attn_implementation=sdpa
combo oft_q3 $Q3 framework.name=QwenOFT

say "4. judge, STARVLA_FAST=exact"
python3 bench/bench.py check --tree starVLA_next --config switches/unified_exact.json --out $R/judge_exact 2>&1 | tail -n 3
say "done"
