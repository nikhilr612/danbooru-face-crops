#!/bin/bash
# Runs stage 1 over the g batches in order; resumable (full_run.py skips ids already in the batch's jsonl). Stop by killing this script's python child and the script.
cd "$(dirname "$0")/.."
for k in $(seq -w 0 19); do
  .envs/landmarks/bin/python build/full_run.py build_data/ids_g_batch$k.json build_data/run_g_batch$k.jsonl 100000 28 2>&1 | grep -v "load checkpoint\|CUDA" >> build_data/full_run.log
done
echo ALL_DONE >> build_data/full_run.log
