#!/bin/bash
# The final tree (starVLA_r15) in one 2-GPU allocation: the judge, the rows that the paths added since their last
# measurement can change, and the distance checks again (now under the trainer's autocast, compared in chunks).
#   1. distance of every rounding-level path from the default step, by module, with yardsticks that are not ours
#   2. the frozen judge: `exact` (gates 0-2), and the three-process final check of `all`
#   3. headline recipe: upstream, exact, all, and `all` of the tree measured in session 4, alternating
#   4. QwenDiscreteDiffusion (its config was wrong in session 6)
#   5. PI0 and PI05 with worker_examples and openpi_sdpa
#   6. the other frameworks whose `all` gained condition_once
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_r15
OLD=$WS/starVLA_next
R=runs/r15
C=$WS/coverage/configs
Q3=$A/models/Qwen3-VL-4B-Instruct
Q25=$A/models/Qwen2.5-VL-3B-Instruct
cd $WS && mkdir -p $R/distance $R/cov
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1); load $(cut -d' ' -f1-3 /proc/loadavg)"

say "1. distance of rounding-level paths"
( source $A/env.sh
D=$WS/$R/distance
dist() { CUDA_VISIBLE_DEVICES=$1 python coverage/rounding_distance.py $T $3 $4 $5 $6 $D/$2.json > $D/$2.log 2>&1; }
dist 0 sdpa_gr00t_q25 QwenGR00T $Q25 flash_attention_2 sdpa & dist 1 sdpa_oft_q25 QwenOFT $Q25 flash_attention_2 sdpa & wait
dist 0 eager_gr00t_q25 QwenGR00T $Q25 sdpa attn:eager & dist 1 eager_oft_q25 QwenOFT $Q25 sdpa attn:eager & wait
dist 0 relaxed_gr00t_q25 QwenGR00T $Q25 sdpa vision_attn_relaxed & dist 1 relaxed_oft_q25 QwenOFT $Q25 sdpa vision_attn_relaxed & wait
dist 0 compilevision_gr00t_q25 QwenGR00T $Q25 sdpa compile_vision & dist 1 compilevision_oft_q25 QwenOFT $Q25 sdpa compile_vision & wait
dist 0 compile_gr00t_q25 QwenGR00T $Q25 sdpa compile_lm & dist 1 compile_oft_q25 QwenOFT $Q25 sdpa compile_lm & wait
dist 0 once_gr00t_q3 QwenGR00T $Q3 sdpa condition_once & dist 1 once_pi_q3 QwenPI $Q3 sdpa condition_once & wait
PI=$T/examples/simBenchmarks/LIBERO/train_files/openpi
pi() {  # pi <gpu> <framework> <yaml> <checkpoint dir> <name> <paths>
    CHECK_CONFIG=$PI/$3 CHECK_SET="framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model" CHECK_ACTION_STEPS=8 \
    CHECK_STATE_DIM=8 CHECK_CHECKPOINT=$A/models/openpi/$4/fp32/model.safetensors dist $1 $5 $2 - eager $6; }
pi 0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla views_pi0 unused_views & pi 1 PI05 pi05_libero_8gpu.yaml pi05_base_starvla views_pi05 unused_views & wait
pi 0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla sdpa_pi0 openpi_sdpa & pi 1 PI05 pi05_libero_8gpu.yaml pi05_base_starvla sdpa_pi05 openpi_sdpa & wait
tail -q -n 1 $D/*.log | cut -c1-700 )

say "2. judge"
python3 bench/bench.py check --tree starVLA_r15 --config switches/unified_exact.json --out $R/judge_exact 2>&1 | tail -n 3
touch task/final-evaluation
python3 bench/bench.py check --final --tree starVLA_r15 --config switches/unified_all.json --out $R/final_all 2>&1 | tail -n 3
rm -f task/final-evaluation

cov() {  # cov <name> <tag> <tree> <fast> <model> <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    say "$name / $tag (load $(cut -d' ' -f1 /proc/loadavg))"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model --set "$@" 2>&1 | tail -n 1 | cut -c1-400
}
runs() {  # runs <name> <model> "<tags>" <overrides...>: ref = upstream, *_old = the session 4 tree, the others = STARVLA_FAST levels
    local name=$1 model=$2 tags=$3; shift 3
    for tag in $tags; do
        case $tag in
            ref*) cov $name $tag targets/reference "" $model "$@" ;;
            all_tf32) cov $name all_tf32 $T all,tf32 $model "$@" ;;
            all_old*) cov $name $tag $OLD all $model "$@" ;;
            all*) cov $name $tag $T all $model "$@" ;;
            *) cov $name $tag $T $tag $model "$@" ;;
        esac
    done
}
say "3. headline recipe, this tree and the session 4 tree alternating"
runs oft_q3 $Q3 "ref exact all all_old all2 all_old2 all_tf32 ref2" framework.name=QwenOFT

say "4. QwenDiscreteDiffusion"
runs dd_q3 $Q3 "ref exact all all_tf32" config_yaml=$C/discrete_diffusion.yaml framework.name=QwenDiscreteDiffusion
say "5. PI0 and PI05 (PaliGemma, openpi recipe on LIBERO, converted base checkpoints)"
P=$T/examples/simBenchmarks/LIBERO/train_files/openpi
openpi() {  # openpi <framework> <yaml> <checkpoint dir>
    echo config_yaml=$P/$2 framework.name=$1 framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model \
        trainer.pretrained_checkpoint=$A/models/openpi/$3/fp32/model.safetensors datasets.vla_data.data_mix=openpi_libero_all trainer.eval_interval=100000000
}
runs pi0 $Q3 "ref exact all all_tf32" $(openpi PI0 pi0_libero_8gpu.yaml pi0_base_starvla)
runs pi05 $Q3 "ref exact all all_tf32" $(openpi PI05 pi05_libero_8gpu.yaml pi05_base_starvla)
say "6. the other frameworks whose all level gained condition_once"
runs piv3_q3 $Q3 "ref exact all all_tf32" framework.name=QwenPI_v3
runs dual_q3 $Q3 "ref exact all all_tf32" config_yaml=$C/qwendual.yaml framework.name=QwenDual
runs langforce_q3 $Q3 "ref exact all all_tf32" config_yaml=$C/langforce.yaml framework.name=LangForce
runs gr00t_q25_fa2 $Q25 "ref exact all all_tf32" framework.name=QwenGR00T
say "done"
