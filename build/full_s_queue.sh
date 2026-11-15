#!/bin/bash
# Rating-s pipeline: stage 1 (detect), stage 2 (frame crops) and stage 3 (parse + background score), pipelined over the 10 s batches.
cd "$(dirname "$0")/.."
(
  for k in $(seq 0 9); do
    n=$(printf %02d $k)
    [ -f build_data/run_s_batch$n.done ] || { .envs/landmarks/bin/python build/full_run.py build_data/ids_s_batch$n.json build_data/run_s_batch$n.jsonl 100000 28 2>&1 | grep -v "load checkpoint\|CUDA" >> build_data/full_run_s.log; touch build_data/run_s_batch$n.done; }
  done
) &
(
  for k in $(seq 0 9); do
    n=$(printf %02d $k)
    while [ ! -f build_data/run_s_batch$n.done ]; do sleep 60; done
    [ -f build_data/crops/s${n}_status.json ] || .envs/metrics/bin/python -I build/full_crop.py s$n >> build_data/full_crop_s.log 2>&1
  done
) &
(
  for k in $(seq 0 9); do
    n=$(printf %02d $k)
    while [ ! -f build_data/crops/s${n}_status.json ]; do sleep 60; done
    [ -f build_data/crops/s${n}_bg.npy ] || .envs/seg/bin/python -I build/full_bg.py s$n >> build_data/full_bg_s.log 2>&1
  done
  echo ALL_DONE >> build_data/full_bg_s.log
) &
wait
