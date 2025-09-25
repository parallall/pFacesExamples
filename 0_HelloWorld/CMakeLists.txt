cmake_minimum_required(VERSION 3.10)

# Name to the project
project(pFaces-Example)

# Include CMake configs for pFaces
if("$ENV{PFACES_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "Please make sure pFaces is installed correctly.")
else()
  include($ENV{PFACES_SDK_ROOT}/pfaces_kernel.cmake)
endif()

# Kernel name and its output path
set(PFACES_KERNEL example)
set(PFACES_KERNEL_OUT_PATH "${PROJECT_SOURCE_DIR}/kernel-pack")

# Add the kernel as a shared lib
add_library(${PFACES_KERNEL} SHARED
  ${PROJECT_SOURCE_DIR}/kernel-driver/example.cpp
)
target_include_directories(${PFACES_KERNEL} PUBLIC
    ${PROJECT_SOURCE_DIR}/kernel-driver
)

# Include link configs for the kernel.
include($ENV{PFACES_SDK_ROOT}/pfaces_link.cmake)
