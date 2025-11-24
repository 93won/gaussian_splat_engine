#!/bin/bash

# Usage: ./run.sh [model_name] [--no-build]
# Example: ./run.sh truck
#          ./run.sh bonsai
#          ./run.sh truck --no-build

# Default model if not specified
MODEL=${1:-bonsai}
BUILD=true

# Check for --no-build flag
for arg in "$@"; do
    if [ "$arg" == "--no-build" ] || [ "$arg" == "-n" ]; then
        BUILD=false
    fi
done

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Paths
BUILD_DIR="${SCRIPT_DIR}/build"
DATA_DIR="${SCRIPT_DIR}/sample_data/models/${MODEL}"
PLY_FILE="${DATA_DIR}/point_cloud/iteration_30000/point_cloud.ply"
CAMERAS_FILE="${DATA_DIR}/cameras.json"

# Check if model exists
if [ ! -f "${PLY_FILE}" ]; then
    echo "Error: Model '${MODEL}' not found!"
    echo "Expected: ${PLY_FILE}"
    echo ""
    echo "Available models:"
    ls -1 "${SCRIPT_DIR}/sample_data/models/" 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Usage: $0 [model_name] [--no-build]"
    echo "Example: $0 truck"
    echo "         $0 bonsai"
    echo "         $0 truck --no-build  (skip rebuild)"
    exit 1
fi

if [ ! -f "${CAMERAS_FILE}" ]; then
    echo "Error: cameras.json not found for model '${MODEL}'!"
    echo "Expected: ${CAMERAS_FILE}"
    exit 1
fi

# Build if needed
if [ "$BUILD" == "true" ]; then
    echo "=========================================="
    echo "Building test_rasterizer_3d..."
    echo "=========================================="

    cd "${BUILD_DIR}" || exit 1
    make test_rasterizer_3d -j8

    if [ $? -ne 0 ]; then
        echo "Error: Build failed!"
        exit 1
    fi
    cd "${SCRIPT_DIR}" || exit 1
else
    echo "=========================================="
    echo "Skipping build (using existing binary)"
    echo "=========================================="
fi

echo ""
echo "=========================================="
echo "Running with model: ${MODEL}"
echo "=========================================="
echo "PLY file: ${PLY_FILE}"
echo "Cameras:  ${CAMERAS_FILE}"
echo "=========================================="
echo ""

cd "${SCRIPT_DIR}" || exit 1
./build/test_rasterizer_3d "${PLY_FILE}" "${CAMERAS_FILE}"
