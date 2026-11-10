#!/bin/sh
set -e

# build the driver if not already built
[ ! -f ./kernel-pack/example.driver ] && sh ./build.sh

echo "[0_HelloWorld] Running with default configuration..."
pfaces -CG -d 1 -k example -v0

echo "[0_HelloWorld] Running with example.cfg + profiling..."
pfaces -CG -d 1 -p -k example -cfg example.cfg

echo "[0_HelloWorld] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k example -cfg example.cfg
