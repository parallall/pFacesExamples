#!/bin/sh
set -e

[ ! -f ./kernel-pack/heat2d.driver ] && sh ./build.sh

echo "[4_Stencil2D_Heat] Running with default configuration..."
pfaces -CG -d 1 -k heat2d

echo "[4_Stencil2D_Heat] Running with example.cfg + profiling..."
pfaces -CG -d 1 -p -k heat2d -cfg example.cfg

echo "[4_Stencil2D_Heat] Running with host-pinned shared memory (-hm)..."
pfaces -CG -d 1 -p -hm -k heat2d -cfg example.cfg
