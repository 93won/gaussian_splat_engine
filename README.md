# Gaussian Splat Engine

Modern C++ implementation of 3D Gaussian Splatting with CUDA acceleration.

## Overview

A complete, from-scratch implementation of Gaussian Splatting for real-time rendering and training. Built with modern C++17 and CUDA for maximum performance.

## Features

- **CUDA-accelerated Rasterization** - Tile-based rendering for real-time performance
- **Modern C++ Design** - Clean architecture with C++17 features
- **Python Bindings** - Easy-to-use Python API with pybind11
- **Training Pipeline** - Full forward/backward implementation with optimization

## Project Status

🚧 **Work in Progress**

Currently implementing:
- [x] Project structure
- [ ] PLY file loader
- [ ] CUDA rasterizer (forward pass)
- [ ] Training pipeline (backward pass)
- [ ] Optimization algorithms (Adam, SGD)
- [ ] Python bindings

## Architecture

Pure rendering and training engine

```
gaussian_splat_engine/
├── src/                    # C++ Core Engine
│   ├── database/           # Gaussian data structures
│   ├── rendering/          # Forward pass (CUDA)
│   ├── training/           # Backward pass (CUDA)
│   ├── optimization/       # Optimizers (Adam, SGD)
│   └── util/               # PLY loader, helpers
│
├── python/                 # Python Bindings
│   ├── gs_engine/          # Python package
│   └── examples/           # Python examples
```

**Design Philosophy:**
- Core engine in C++/CUDA for performance
- Python bindings for easy prototyping
- Modular architecture for flexibility
- Minimal dependencies

## Build Requirements

- CUDA 11.6+
- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+)
- Eigen3
- pybind11 (for Python bindings)
- Python 3.8+ (optional, for Python API)


⚡ Built with performance in mind | 🎨 Designed for clarity | 🚀 Optimized with CUDA
