/**
 * @file      PLYUtils.cpp
 * @brief     PLY file I/O utilities for Gaussian Model
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 */

#include "PLYUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>

namespace ply {

/**
 * @brief Load Gaussians from binary PLY file
 */
bool LoadGaussians(const std::string& ply_path, std::vector<Gaussian>& gaussians) {
    std::ifstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[PLYUtils] Cannot open PLY file: " << ply_path << std::endl;
        return false;
    }
    
    gaussians.clear();
    
    // Read ASCII header
    std::string line;
    int num_vertices = 0;
    bool binary_format = false;
    
    while (std::getline(file, line)) {
        if (line.find("format binary_little_endian") != std::string::npos) {
            binary_format = true;
        }
        if (line.find("element vertex") != std::string::npos) {
            std::istringstream iss(line);
            std::string dummy1, dummy2;
            iss >> dummy1 >> dummy2 >> num_vertices;
        }
        if (line == "end_header") {
            break;
        }
    }
    
    if (!binary_format) {
        std::cerr << "[PLYUtils] Only binary_little_endian PLY format is supported" << std::endl;
        return false;
    }
    
    if (num_vertices <= 0) {
        std::cerr << "[PLYUtils] Invalid number of vertices: " << num_vertices << std::endl;
        return false;
    }
    
    // Read binary data
    // PLY format from 3DGS: x, y, z, nx, ny, nz, 
    //                       f_dc_0, f_dc_1, f_dc_2 (DC - 3 floats)
    //                       f_rest_0 ~ f_rest_44 (SH degree 1-3, 45 floats)
    //                       opacity, scale_0, scale_1, scale_2, 
    //                       rot_0, rot_1, rot_2, rot_3
    struct PLYGaussian {
        float x, y, z;              // position (3)
        float nx, ny, nz;           // normal (3) - not used
        float f_dc_0, f_dc_1, f_dc_2;  // SH DC component (3)
        float f_rest[45];           // SH rest components (45) - we skip these for now
        float opacity;              // opacity (1)
        float scale_0, scale_1, scale_2;  // scale (3)
        float rot_0, rot_1, rot_2, rot_3;  // quaternion (4) - format: w, x, y, z
    };
    
    gaussians.reserve(num_vertices);
    
    for (int i = 0; i < num_vertices; ++i) {
        PLYGaussian ply_gaussian;
        file.read(reinterpret_cast<char*>(&ply_gaussian), sizeof(PLYGaussian));
        
        if (!file) {
            std::cerr << "[PLYUtils] Failed to read Gaussian " << i << " from PLY file" << std::endl;
            break;
        }
        
        Gaussian gaussian;
        gaussian.position = Eigen::Vector3f(ply_gaussian.x, ply_gaussian.y, ply_gaussian.z);
        gaussian.sh_dc = Eigen::Vector3f(ply_gaussian.f_dc_0, ply_gaussian.f_dc_1, ply_gaussian.f_dc_2);
        
        // Copy all 45 SH rest coefficients
        for (int j = 0; j < 45; ++j) {
            gaussian.sh_rest[j] = ply_gaussian.f_rest[j];
        }
        
        gaussian.opacity = ply_gaussian.opacity;
        gaussian.scale = Eigen::Vector3f(ply_gaussian.scale_0, ply_gaussian.scale_1, ply_gaussian.scale_2);
        
        // Quaternion from PLY: (w, x, y, z)
        gaussian.rotation = Eigen::Quaternionf(ply_gaussian.rot_0, ply_gaussian.rot_1, 
                                               ply_gaussian.rot_2, ply_gaussian.rot_3);
        
        gaussians.push_back(gaussian);
    }
    
    file.close();
    
    if (gaussians.empty()) {
        std::cerr << "[PLYUtils] No Gaussians loaded from PLY file" << std::endl;
        return false;
    }
    
    std::cout << "[PLYUtils] Loaded " << gaussians.size() << " Gaussians from " << ply_path << std::endl;
    return true;
}

/**
 * @brief Save Gaussians to binary PLY file
 */
bool SaveGaussians(const std::string& ply_path, const std::vector<Gaussian>& gaussians) {
    if (gaussians.empty()) {
        std::cerr << "[PLYUtils] No Gaussians to save" << std::endl;
        return false;
    }
    
    std::ofstream file(ply_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[PLYUtils] Cannot create PLY file: " << ply_path << std::endl;
        return false;
    }
    
    // Write ASCII header
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << gaussians.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    // Write f_rest headers (45 coefficients for SH degree 3)
    for (int i = 0; i < 45; ++i) {
        file << "property float f_rest_" << i << "\n";
    }
    file << "property float opacity\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "end_header\n";
    
    // Write binary data
    struct PLYGaussian {
        float x, y, z;
        float nx, ny, nz;
        float f_dc_0, f_dc_1, f_dc_2;
        float f_rest[45];  // SH rest - set to zero for now
        float opacity;
        float scale_0, scale_1, scale_2;
        float rot_0, rot_1, rot_2, rot_3;
    };
    
    for (const auto& gaussian : gaussians) {
        PLYGaussian ply_gaussian;
        
        ply_gaussian.x = gaussian.position.x();
        ply_gaussian.y = gaussian.position.y();
        ply_gaussian.z = gaussian.position.z();
        
        // Normal not used, set to zero
        ply_gaussian.nx = 0.0f;
        ply_gaussian.ny = 0.0f;
        ply_gaussian.nz = 0.0f;
        
        ply_gaussian.f_dc_0 = gaussian.sh_dc.x();
        ply_gaussian.f_dc_1 = gaussian.sh_dc.y();
        ply_gaussian.f_dc_2 = gaussian.sh_dc.z();
        
        // Copy all 45 SH rest coefficients
        for (int j = 0; j < 45; ++j) {
            ply_gaussian.f_rest[j] = gaussian.sh_rest[j];
        }
        
        ply_gaussian.opacity = gaussian.opacity;
        
        ply_gaussian.scale_0 = gaussian.scale.x();
        ply_gaussian.scale_1 = gaussian.scale.y();
        ply_gaussian.scale_2 = gaussian.scale.z();
        
        // Quaternion: (w, x, y, z)
        ply_gaussian.rot_0 = gaussian.rotation.w();
        ply_gaussian.rot_1 = gaussian.rotation.x();
        ply_gaussian.rot_2 = gaussian.rotation.y();
        ply_gaussian.rot_3 = gaussian.rotation.z();
        
        file.write(reinterpret_cast<const char*>(&ply_gaussian), sizeof(PLYGaussian));
    }
    
    file.close();
    
    std::cout << "[PLYUtils] Saved " << gaussians.size() << " Gaussians to " << ply_path << std::endl;
    return true;
}

} // namespace ply
