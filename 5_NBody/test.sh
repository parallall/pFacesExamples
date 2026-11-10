#!/bin/sh
set -e

[ ! -f ./kernel-pack/nbody.driver ] && sh ./build.sh

echo "[5_NBody] Running with default configuration..."
pfaces -CG -d 1 -k nbody

echo "[5_NBody] Running with example.cfg + profiling..."
pfaces -CG -d 1 -p -k nbody -cfg example.cfg

echo "[5_NBody] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k nbody -cfg example.cfg
