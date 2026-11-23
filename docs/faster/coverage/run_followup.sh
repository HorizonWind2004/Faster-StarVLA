#!/bin/bash
# Follow-up validation on the final tree (starVLA_next = fast-unified-next, with the flash-attention handling):
#   1. the real-weights bitwise suite again, plus QwenOFT on Qwen2.5-VL with flash-attention
#   2. STARVLA_FAST=all on the two flash-attention recipes that failed before the fix (reference and exact exist)
#   3. the frozen judge at this commit: exact set (gates 0-2), then the final check with everything on
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_next
R=runs/unified
cd $WS && mkdir -p $R/bitwise4
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1)"

say "1. bitwise on real weights"
( source $A/env.sh
B=$WS/$R/bitwise4; Q3=$A/models/Qwen3-VL-4B-Instruct; Q25=$A/models/Qwen2.5-VL-3B-Instruct
bb() { CUDA_VISIBLE_DEVICES=$1 CHECK_DTYPE=$4 CHECK_PATHS=$5 python coverage/gpu_bitwise.py $T $3 $B/$2.json > $B/$2.log 2>&1; }
fw() { CUDA_VISIBLE_DEVICES=$1 python coverage/framework_bitwise.py $T $3 $4 $5 $B/$2.json > $B/$2.log 2>&1; }
E=attn_mask,hidden_only,vision_attn,vision_pos
bb 0 bb_q3 $Q3 bf16 $E & bb 1 bb_q25 $Q25 bf16 $E & wait
bb 0 bb_q25_relaxed_fp32 $Q25 fp32 $E,vision_attn_relaxed & fw 1 fw_oft_q3 QwenOFT $Q3 sdpa & wait
fw 0 fw_gr00t_q3 QwenGR00T $Q3 sdpa & fw 1 fw_pi_q3 QwenPI $Q3 sdpa & wait
fw 0 fw_oft_q25 QwenOFT $Q25 sdpa & fw 1 fw_gr00t_q25 QwenGR00T $Q25 sdpa & wait
fw 0 fw_gr00t_q25_fa2 QwenGR00T $Q25 flash_attention_2 & fw 1 fw_oft_q25_fa2 QwenOFT $Q25 flash_attention_2 & wait
tail -q -n 1 $B/*.log | cut -c1-300 )

cov() {  # cov <name> <tag> <fast> <model> <overrides...>
    local name=$1 tag=$2 fast=$3 model=$4; shift 4
    say "2. $name / $tag"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $T --fast "$fast" --out $R/cov/$name --tag $tag --model $model --set "$@" 2>&1 | tail -n 1
}
Q25=$A/models/Qwen2.5-VL-3B-Instruct
cov oft_q25_fa2 all_fixed all $Q25 framework.name=QwenOFT
cov gr00t_q25_fa2 all_fixed all $Q25 framework.name=QwenGR00T
cov gr00t_q3 all_fixed all $A/models/Qwen3-VL-4B-Instruct framework.name=QwenGR00T   # the earlier run predates the probe

say "3a. judge, STARVLA_FAST=exact"
python3 bench/bench.py check --tree starVLA_next --config switches/unified_exact.json --out $R/final_exact 2>&1 | tail -n 3

say "3b. judge, final check, STARVLA_FAST=all"
touch task/final-evaluation
python3 bench/bench.py check --final --tree starVLA_next --config switches/unified_all.json --out $R/final_all 2>&1 | tail -n 3
rm -f task/final-evaluation
say "done"
