#!/bin/sh
set -e

[ ! -f ./kernel-pack/reduction.driver ] && sh ./build.sh

echo "[3_ParallelReduction] Running with default configuration..."
pfaces -CG -d 1 -k reduction

echo "[3_ParallelReduction] Running with example.cfg + profiling..."
pfaces -CG -d 1 -p -k reduction -cfg example.cfg

echo "[3_ParallelReduction] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k reduction -cfg example.cfg
