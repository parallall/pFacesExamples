#!/bin/sh

# make the example
make

# run the example
pfaces -CG -d 1 -p -k example@./kernel-pack -cfg example.cfg