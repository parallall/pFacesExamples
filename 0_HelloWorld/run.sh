#!/bin/sh
set -e

# build the example
sh ./build.sh

# run the example
pfaces -CG -d 1 -k example -v0

# run the example with the config file
pfaces -CG -d 1 -p -k example -cfg example.cfg
