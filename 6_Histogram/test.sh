#!/bin/sh
set -e

[ ! -f ./kernel-pack/histogram.driver ] && sh ./build.sh

echo "[6_Histogram] Running with default configuration..."
pfaces -CG -d 1 -k histogram

echo "[6_Histogram] Running with example.cfg + profiling..."
pfaces -CG -d 1 -p -k histogram -cfg example.cfg

echo "[6_Histogram] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k histogram -cfg example.cfg
