#!/bin/bash
# PI0 and PI05 again (tree starVLA_r16). In session 8, worker_examples did not apply under the trainer (its stand-in
# did not understand the trainer's config wrapper; fixed), and openpi_sdpa looked faster on PI0 and slower on PI05
# against session 7. Since then: openpi_compile (level all), and openpi_sdpa moved to the precision level (max).
#   1. distance of one training step with openpi_compile from the default step
#   2. step times in one job: upstream, exact, and all / max alternating (max = all + openpi_sdpa + tf32)
WS=$WORK/starvla/ws
A=$WORK/starvla/assets
T=$WS/starVLA_r16
R=runs/r16
Q3=$A/models/Qwen3-VL-4B-Instruct
cd $WS && mkdir -p $R/cov $R/distance
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $T log --oneline | head -1); load $(cut -d' ' -f1-3 /proc/loadavg)"

say "1. distance of openpi_compile"
( source $A/env.sh
D=$WS/$R/distance
PI=$T/examples/simBenchmarks/LIBERO/train_files/openpi
pi() {  # pi <gpu> <framework> <yaml> <checkpoint dir> <name> <paths>
    CUDA_VISIBLE_DEVICES=$1 CHECK_CONFIG=$PI/$3 CHECK_SET="framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model" \
    CHECK_ACTION_STEPS=8 CHECK_STATE_DIM=8 CHECK_CHECKPOINT=$A/models/openpi/$4/fp32/model.safetensors \
    python coverage/rounding_distance.py $T $2 - eager $6 $D/$5.json > $D/$5.log 2>&1; }
pi 0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla compile_pi0 openpi_compile & pi 1 PI05 pi05_libero_8gpu.yaml pi05_base_starvla compile_pi05 openpi_compile & wait
tail -q -n 1 $D/*.log | cut -c1-700 )

cov() {  # cov <name> <tag> <tree> <fast> <model> <overrides...>
    local name=$1 tag=$2 tree=$3 fast=$4 model=$5; shift 5
    say "$name / $tag (load $(cut -d' ' -f1 /proc/loadavg))"
    NO_INJECT=1 python3 coverage/time_cfg.py --tree $tree --fast "$fast" --out $R/cov/$name --tag $tag --model $model --set "$@" 2>&1 | tail -n 1 | cut -c1-400
}
runs() {  # runs <name> <model> "<tags>" <overrides...>: a tag is a level, optionally followed by a digit (a repeat)
    local name=$1 model=$2 tags=$3; shift 3
    for tag in $tags; do
        case $tag in
            ref*) cov $name $tag targets/reference "" $model "$@" ;;
            *) cov $name $tag $T ${tag%[0-9]} $model "$@" ;;
        esac
    done
}
say "2. step times"
P=$T/examples/simBenchmarks/LIBERO/train_files/openpi
openpi() {  # openpi <framework> <yaml> <checkpoint dir>
    echo config_yaml=$P/$2 framework.name=$1 framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model \
        trainer.pretrained_checkpoint=$A/models/openpi/$3/fp32/model.safetensors datasets.vla_data.data_mix=openpi_libero_all trainer.eval_interval=100000000
}
runs pi0 $Q3 "ref exact all max all2 max2" $(openpi PI0 pi0_libero_8gpu.yaml pi0_base_starvla)
runs pi05 $Q3 "ref exact all max all2 max2" $(openpi PI05 pi05_libero_8gpu.yaml pi05_base_starvla)
say "done"
