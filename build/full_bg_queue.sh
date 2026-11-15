#!/bin/bash
# Stage 3 over the g batches in order, each as soon as stage 2 has finished it.
cd "$(dirname "$0")/.."
for k in $(seq 0 19); do
  t=build_data/crops/g$(printf %02d $k)
  while [ ! -f ${t}_status.json ]; do sleep 60; done
  [ -f ${t}_bg.npy ] || .envs/seg/bin/python -I build/full_bg.py $k >> build_data/full_bg.log 2>&1
done
echo ALL_DONE >> build_data/full_bg.log
