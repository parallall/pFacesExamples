#!/bin/sh
set -e

[ ! -f ./kernel-pack/vecadd.driver ] && sh ./build.sh

echo "[1_VectorAdd] Running with default configuration..."
pfaces -CG -d 1 -k vecadd

echo "[1_VectorAdd] Running with example.cfg + profiling..."
pfaces -CG -d 1 -p -k vecadd -cfg example.cfg

echo "[1_VectorAdd] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k vecadd -cfg example.cfg
