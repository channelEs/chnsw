#!/bin/bash

set -e

mkdir -p build

cmake -B build \
      -S . \
      -DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake

cmake --build build