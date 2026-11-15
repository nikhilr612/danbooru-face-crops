#!/bin/bash
# Stage 4 over finished batches (g00..g19 then s00..s09 as stage 3 finishes them).
cd "$(dirname "$0")/.."
for t in $(seq -f "g%02g" 0 19) $(seq -f "s%02g" 0 9); do
  while [ ! -f build_data/crops/${t}_bg.npy ]; do sleep 60; done
  [ -f build_data/feats/${t}_wd.npy ] || .envs/metrics/bin/python -I build/full_feats.py $t >> build_data/full_feats.log 2>&1
done
echo ALL_DONE >> build_data/full_feats.log
