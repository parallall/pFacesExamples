#!/bin/sh

# make the example
make

# run the example
pfaces -CG -d 1 -k example -v0

# run the example with config gile
pfaces -CG -d 1 -p -k example -cfg example.cfg