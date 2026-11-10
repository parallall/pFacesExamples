cmake_minimum_required(VERSION 3.22...3.31)

project(pFaces-Example-Histogram)

if("$ENV{PFACES_SDK_ROOT}" STREQUAL "")
  message(FATAL_ERROR "Please make sure pFaces is installed correctly (PFACES_SDK_ROOT is not set).")
else()
  include($ENV{PFACES_SDK_ROOT}/pfaces_kernel.cmake)
endif()

set(PFACES_KERNEL histogram)
set(PFACES_KERNEL_OUT_PATH "${PROJECT_SOURCE_DIR}/kernel-pack")

add_library(${PFACES_KERNEL} SHARED
  ${PROJECT_SOURCE_DIR}/kernel-driver/histogram.cpp
)
target_include_directories(${PFACES_KERNEL} PUBLIC
  ${PROJECT_SOURCE_DIR}/kernel-driver
)

include($ENV{PFACES_SDK_ROOT}/pfaces_link.cmake)
