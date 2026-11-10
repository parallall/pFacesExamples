#!/bin/sh
set -e

BUILDTYPE=Release
CLEAN_BUILD=true

if [ "$CLEAN_BUILD" = true ]; then
  rm -rf kernel-pack/*.driver build
fi

mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE="$BUILDTYPE"
cmake --build build --config "$BUILDTYPE"
