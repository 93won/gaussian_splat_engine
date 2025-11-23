#!/bin/bash

set -e  # Exit on any error

echo "=========================================="
echo " Building Gaussian Splat Engine"
echo "=========================================="
echo ""

# Get number of CPU cores for parallel compilation (use half of available cores)
NPROC=$(($(nproc) / 2))
if [ $NPROC -lt 1 ]; then
    NPROC=1
fi
echo "Using $NPROC cores for compilation (half of available)"
echo ""

# Create build directory
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

cd build

# CMake configuration
echo "Running CMake..."
cmake ..

if [ $? -ne 0 ]; then
    echo "CMake configuration failed!"
    exit 1
fi

# Build
echo "Building..."
make -j$NPROC

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo " Build completed successfully!"
echo "=========================================="
echo ""
echo "Executables:"
echo "  - build/test_ply_loader"
echo ""
echo "Libraries:"
echo "  - build/libgaussian_splat_engine.a"
echo ""
