#!/bin/sh
set -e

[ ! -f ./kernel-pack/matmul.driver ] && sh ./build.sh

echo "[2_MatrixMul] Running with default configuration..."
pfaces -CG -d 1 -k matmul

echo "[2_MatrixMul] Running with example.cfg + on-the-fly tuning (-t) + profiling (-p)..."
pfaces -CG -d 1 -t -p -k matmul -cfg example.cfg

echo "[2_MatrixMul] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k matmul -cfg example.cfg
