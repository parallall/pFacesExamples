#!/bin/sh
set -e

# run the example without config
pfaces -CG -d 1 -k example -v0

# run the example with the config file
pfaces -CG -d 1 -p -k example -cfg example.cfg
