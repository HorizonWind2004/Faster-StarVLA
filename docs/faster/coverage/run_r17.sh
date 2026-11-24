#!/bin/bash
# The statistical gate on a framework whose head is fp32 (QwenPI), and the judge on the final tree (starVLA_final).
#   1. QwenPI: upstream on six seeds, then STARVLA_FAST=all and =max on seed 0 against the seed-to-seed spread
#      (coverage/stat_gate.py: the judge's recorder, trajectory case and 3-sigma criterion with another framework)
#   2. the frozen judge on the headline recipe: all (gates 0-3 against the frozen timing), then exact (gates 0-2)
WS=$WORK/starvla/ws
R=runs/r17
cd $WS && mkdir -p $R
say() { echo "=== $(date +%H:%M:%S) $*"; }
say "tree $(git -C $WS/starVLA_final log --oneline | head -1); load $(cut -d' ' -f1-3 /proc/loadavg)"
say "0. distance of openpi_compile (failed in session 9: the script could not compare steps whose sets of gradients differ)"
( A=$WORK/starvla/assets; T=$WS/starVLA_final; D=$WS/$R/distance; mkdir -p $D
source $A/env.sh
PI=$T/examples/simBenchmarks/LIBERO/train_files/openpi
pi() {  # pi <gpu> <framework> <yaml> <checkpoint dir> <name> <paths>
    CUDA_VISIBLE_DEVICES=$1 CHECK_CONFIG=$PI/$3 CHECK_SET="framework.tokenizer.model_path=$A/models/paligemma/paligemma_tokenizer.model" \
    CHECK_ACTION_STEPS=8 CHECK_STATE_DIM=8 CHECK_CHECKPOINT=$A/models/openpi/$4/fp32/model.safetensors \
    python coverage/rounding_distance.py $T $2 - eager $6 $D/$5.json > $D/$5.log 2>&1; }
pi 0 PI0 pi0_libero_8gpu.yaml pi0_base_starvla compile_pi0 openpi_compile & pi 1 PI05 pi05_libero_8gpu.yaml pi05_base_starvla compile_pi05 openpi_compile & wait
tail -q -n 1 $D/*.log | cut -c1-900 )
say "1. statistical gate, QwenPI"
python3 coverage/stat_gate.py QwenPI $R/stat_pi starVLA_final all max 2>&1 | grep -v Warning | tail -n 16 | cut -c1-600
say "2. judge on the final tree: $(git -C $WS/starVLA_final log --oneline | head -1)"
python3 bench/bench.py check --tree starVLA_final --config switches/unified_all.json --out $R/judge_all 2>&1 | tail -n 4
python3 bench/bench.py check --tree starVLA_final --config switches/unified_exact.json --out $R/judge_exact 2>&1 | tail -n 3
say "done"
