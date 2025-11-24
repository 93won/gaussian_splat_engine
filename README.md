# Gaussian Splat Engine

Modern C++ implementation of 3D Gaussian Splatting with CUDA acceleration.

## Overview

An implementation of 3D Gaussian Splatting for educational and research purposes.Built with modern C++17, following the algorithm from the original paper with implementation details referenced from the official open-source code.

## Features

- **CUDA-accelerated Rasterization** - Tile-based rendering for real-time performance
- **Modern C++ Design** - Clean architecture with C++17 features
- **Training Pipeline** - Full forward/backward implementation with optimization

## Project Status

🚧 **Work in Progress**

Currently implementing:
- [x] Project structure
- [x] PLY file loader
- [x] Visualization (1) (3D Color Points only)
- [x] CPU rasterizer (for test)
- [ ] CUDA rasterizer (forward pass)
- [ ] Visualization (2) (Rendering)
- [ ] Training pipeline (backward pass)
- [ ] Optimization algorithms (Adam, SGD)

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
```

**Design Philosophy:**
- Core engine in C++/CUDA for performance
- Modular architecture for flexibility
- Minimal dependencies

## Build Requirements

- CUDA 11.6+
- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+)
- Eigen3


⚡ Built with performance in mind | 🎨 Designed for clarity | 🚀 Optimized with CUDA

---

## License & Attribution

This project is for **research and educational purposes only**.

### Original 3D Gaussian Splatting

This implementation follows the algorithm described in:

**3D Gaussian Splatting for Real-Time Radiance Field Rendering**  
Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis  
ACM Transactions on Graphics, 2023  
[Project Page](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf)

Implementation details referenced from the official open-source code:  
[graphdeco-inria/diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)

### Citation

If you use this code in your research, please cite the original work:

```bibtex
@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

### This Implementation

This repository provides a CPU-based educational implementation for learning purposes.

**Author:** Seungwon Choi (csw3575@snu.ac.kr)  
**Purpose:** Research benchmark for monocular Gaussian splatting SLAM
