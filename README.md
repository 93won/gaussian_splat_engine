# Gaussian Splat Engine

Modern C++ implementation of 3D Gaussian Splatting with CUDA acceleration.

## Overview

A complete, from-scratch implementation of Gaussian Splatting for real-time rendering and training. Built with modern C++17 and CUDA for maximum performance.

## Features

- **CUDA-accelerated Rasterization** - Tile-based rendering for real-time performance
- **Modern C++ Design** - Clean architecture with C++17 features
- **Pangolin Viewer** - Interactive 3D visualization
- **Training Pipeline** - Full forward/backward implementation (coming soon)

## Project Status

🚧 **Work in Progress**

Currently implementing:
- [x] Project structure
- [ ] PLY file loader
- [ ] CUDA rasterizer (forward pass)
- [ ] Pangolin viewer
- [ ] Training pipeline
- [ ] Optimization algorithms

## Build Requirements

- CUDA 11.6+
- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+)
- Eigen3
- Pangolin
- OpenGL

## Quick Start

```bash
# Clone repository
git clone git@github.com:93won/gaussian_splat_engine.git
cd gaussian_splat_engine


⚡ Built with performance in mind | 🎨 Designed for clarity | 🚀 Optimized with CUDA
