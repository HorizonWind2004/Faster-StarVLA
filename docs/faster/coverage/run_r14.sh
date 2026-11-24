#!/bin/bash
# condition_once, unused_views, the distance of the rounding-level paths, and PI0 / PI05 (tree starVLA_r14), in one
# 2-GPU allocation.
#   1. how far a step with a rounding-level fast path is from the default step, by module (real weights, one GPU each)
#   2. step time of the DiT-head frameworks with the new `all` (condition_once)
#   3. PI0 and PI05 (PaliGemma, openpi recipe on LIBERO, converted base checkpoints): profile of upstream, step times
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_r14
R=runs/r14
Q3=$A/models/Qwen3-VL-4B-Instruct
Q25=$A/models/Qwen2.5-VL-3B-Instruct
cd $WS && mkdir -p $R/distance $R/cov
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1)"

say "1. distance of rounding-level paths"
( source $A/env.sh
D=$WS/$R/distance
dist() { CUDA_VISIBLE_DEVICES=$1 python coverage/rounding_distance.py $T $3 $4 $5 $6 $D/$2.json > $D/$2.log 2>&1; }
dist 0 sdpa_gr00t_q25 QwenGR00T $Q25 flash_attention_2 sdpa & dist 1 sdpa_oft_q25 QwenOFT $Q25 flash_attention_2 sdpa & wait
dist 0 relaxed_gr00t_q25 QwenGR00T $Q25 sdpa vision_attn_relaxed & dist 1 relaxed_oft_q25 QwenOFT $Q25 sdpa vision_attn_relaxed & wait
dist 0 compile_gr00t_q25 QwenGR00T $Q25 sdpa compile_lm & dist 1 compile_oft_q25 QwenOFT $Q25 sdpa compile_lm & wait
dist 0 once_gr00t_q3 QwenGR00T $Q3 sdpa condition_once & dist 1 once_pi_q3 QwenPI $Q3 sdpa condition_once & wait
PI=$T/examples/simBenchmarks/LIBERO/train_files/openpi
pi() { CHECK_CONFIG=$PI/$3 CHECK_SET="framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model" CHECK_ACTION_STEPS=8 \
       CHECK_STATE_DIM=8 CHECK_CHECKPOINT=$A/models/openpi/$4/fp32/model.safetensors dist $1 views_$5 $2 - eager unused_views; }
pi 0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla pi0 & pi 1 PI05 pi05_libero_8gpu.yaml pi05_base_starvla pi05 & wait
tail -q -n 1 $D/*.log | cut -c1-900 )

cov() {  # cov <name> <tag> <tree> <fast> <model> <extra time_cfg args> -- <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    local extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
    say "$name / $tag"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model "${extra[@]}" --set "$@" 2>&1 | tail -n 1 | cut -c1-400
}
runs() {  # runs <name> <model> "<tags>" <overrides...>: ref = upstream, the others = STARVLA_FAST levels
    local name=$1 model=$2 tags=$3; shift 3
    for tag in $tags; do
        case $tag in
            ref) cov $name ref targets/reference "" $model -- "$@" ;;
            all_tf32) cov $name all_tf32 $T all,tf32 $model -- "$@" ;;
            *) cov $name $tag $T $tag $model -- "$@" ;;
        esac
    done
}
say "2. DiT-head frameworks"
runs gr00t_q3 $Q3 "ref exact all all_tf32" framework.name=QwenGR00T
runs pi_q3 $Q3 "ref exact all all_tf32" framework.name=QwenPI

say "3. PI0 and PI05"
P=$T/examples/simBenchmarks/LIBERO/train_files/openpi
openpi() {  # openpi <name> <framework> <yaml> <checkpoint dir>
    echo config_yaml=$P/$3 framework.name=$2 framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model \
        trainer.pretrained_checkpoint=$A/models/openpi/$4/fp32/model.safetensors datasets.vla_data.data_mix=openpi_libero_all trainer.eval_interval=100000000
}
cov pi0 prof_ref targets/reference "" $Q3 --cprofile 20 -- $(openpi pi0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla)
runs pi0 $Q3 "ref exact all all_tf32" $(openpi pi0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla)
runs pi05 $Q3 "ref exact all" $(openpi pi05 PI05 pi05_libero_8gpu.yaml pi05_base_starvla)
say "4. where the time goes with all fast paths (cProfile, kernels launched blocking)"
cov pi_q3 prof_all $T all $Q3 --cprofile 20 -- framework.name=QwenPI
cov gr00t_q3 prof_all $T all $Q3 --cprofile 20 -- framework.name=QwenGR00T
cov pi0 prof_all $T all $Q3 --cprofile 20 -- $(openpi pi0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla)
say "done"
