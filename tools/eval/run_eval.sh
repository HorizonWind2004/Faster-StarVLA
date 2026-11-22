#!/bin/bash
# LIBERO evaluation (4 suites x 10 tasks x TRIALS episodes) of one starVLA checkpoint on the GPUs of this node:
#
#   CKPT=<run>/checkpoints/steps_N_pytorch_model.pt LIBERO_HOME=<LIBERO clone> LIBERO_PYTHON=<python of the LIBERO env> \
#     [TRIALS=50] [CLIENTS_PER_GPU=14] [OUT=<dir>] tools/eval/run_eval.sh
#
# Run from the repository root in the starVLA environment. One batching policy server per GPU
# (tools/eval/batched_server.py: concurrent requests of the simulation clients become one predict_action call) and
# CLIENTS_PER_GPU simulation clients per GPU; the simulation is CPU work, so size CLIENTS_PER_GPU by the CPU cores.
# The (task, episode) pairs of a suite are split across the clients; each episode's initial state is fixed by its
# index, so the split does not change the test. Per-episode results: <OUT>/<suite>/w<i>/episodes.jsonl.
set -eo pipefail
: ${CKPT:?set CKPT} ${LIBERO_HOME:?set LIBERO_HOME} ${LIBERO_PYTHON:?set LIBERO_PYTHON}
ROOT=$(pwd); E=$ROOT/tools/eval
RUN=$(dirname $(dirname $CKPT)); STEP=$(basename $CKPT .pt)
OUT=${OUT:-$RUN/results_std/$STEP}; mkdir -p $OUT
NG=$(nvidia-smi -L | wc -l); C=${CLIENTS_PER_GPU:-14}; W=$((NG * C))
BASE_PORT=$((31000 + ($$ % 500) * 10))
echo "[eval] $(hostname) ckpt=$CKPT gpus=$NG clients=$W trials=${TRIALS:-50} out=$OUT $(date)"
PIDS=()
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$ROOT python $E/batched_server.py --ckpt_path $CKPT --port $((BASE_PORT + g)) --use_bf16 > $OUT/server$g.log 2>&1 &
  PIDS+=($!)
done
trap 'kill ${PIDS[*]} 2>/dev/null' EXIT
for g in $(seq 0 $((NG - 1))); do
  for i in $(seq 1 120); do
    (echo > /dev/tcp/127.0.0.1/$((BASE_PORT + g))) 2>/dev/null && break
    kill -0 ${PIDS[$g]} 2>/dev/null || { echo "[eval] server $g died"; tail -20 $OUT/server$g.log; exit 1; }
    sleep 5
  done
done
echo "[eval] servers up $(date +%T)"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$LIBERO_HOME/libero/libero}
CLIENTS=()
for w in $(seq 0 $((W - 1))); do
  g=$((w % NG))
  (for suite in ${SUITES:-libero_spatial libero_object libero_goal libero_10}; do
     CUDA_VISIBLE_DEVICES=$g MUJOCO_EGL_DEVICE_ID=$g PYTHONPATH=$LIBERO_HOME:$ROOT \
       $LIBERO_PYTHON $E/eval_libero_std.py \
       --args.pretrained-path $CKPT --args.host 127.0.0.1 --args.port $((BASE_PORT + g)) \
       --args.task-suite-name $suite --args.num-trials-per-task ${TRIALS:-50} --args.no-save-video \
       --args.unit-stride $W --args.unit-offset $w --args.video-out-path $OUT/$suite/w$w
   done) > $OUT/client$w.log 2>&1 &
  CLIENTS+=($!)
done
wait "${CLIENTS[@]}" || true
python3 $E/summarize.py $OUT
echo "[eval] done $(date)"
