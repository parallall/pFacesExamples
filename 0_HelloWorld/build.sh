#!/bin/sh

# CMake cofiguration
BUILDTYPE=Release
CLEAN_BUILD=true

# remove old build binaries
if [ $CLEAN_BUILD = true ]
then 
  rm -rf kernel-pack/*.driver
  rm -rf build
fi

# building ...
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=$BUILDTYPE
cmake --build . --config $BUILDTYPE
cd ..
