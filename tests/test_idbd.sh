#!/bin/bash
# Numerics on the host, then a compile check of the kernel. The device/host
# parity run needs a GPU; without one the compile is still the useful signal.
set -e
cd "$(dirname "$0")/.."

${CC:-cc} -O2 -Wall tests/test_idbd.c -lm -o build_test_idbd
./build_test_idbd

if command -v nvcc >/dev/null 2>&1; then
    nvcc -std=c++17 -arch=sm_80 -Isrc tests/test_idbd.cu -o build_test_idbd_cu
    if nvidia-smi >/dev/null 2>&1; then
        ./build_test_idbd_cu
    else
        echo "ok idbd kernel compiles (no GPU present, parity run skipped)"
    fi
else
    echo "skip: nvcc not found, kernel compile check not run"
fi
