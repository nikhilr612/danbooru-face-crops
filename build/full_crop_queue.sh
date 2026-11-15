#!/bin/bash
# Stage 2 over the g batches in order; skips batches whose output exists.
cd "$(dirname "$0")/.."
for k in $(seq 0 19); do
  [ -f build_data/crops/g$(printf %02d $k)_status.json ] && continue
  .envs/metrics/bin/python -I build/full_crop.py $k >> build_data/full_crop.log 2>&1
done
echo ALL_DONE >> build_data/full_crop.log
