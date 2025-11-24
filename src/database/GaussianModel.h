/**
 * @file      GaussianModel.h
 * @brief     Gaussian primitive data structure and model
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 */

#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

/**
 * @brief Single Gaussian primitive data structure
 */
struct Gaussian {
    Eigen::Vector3f position;       // 3D position (x, y, z)
    Eigen::Vector3f scale;          // Scale (sx, sy, sz)
    Eigen::Quaternionf rotation;    // Rotation as quaternion (w, x, y, z)
    float opacity;                  // Opacity (0~1)
    
    // Spherical Harmonics for view-dependent color
    Eigen::Vector3f sh_dc;          // DC component (degree 0) - base color
    float sh_rest[45];              // Higher order SH coefficients (degree 1-3)
                                    // Layout: [R_1..R_15, G_1..G_15, B_1..B_15]
                                    // Degree 1: 3 coeffs per channel (9 total)
                                    // Degree 2: 5 coeffs per channel (15 total)
                                    // Degree 3: 7 coeffs per channel (21 total)
    
    Gaussian()
        : position(Eigen::Vector3f::Zero())
        , scale(Eigen::Vector3f::Ones())
        , rotation(Eigen::Quaternionf::Identity())
        , opacity(1.0f)
        , sh_dc(Eigen::Vector3f::Zero())
    {
        for (int i = 0; i < 45; ++i) {
            sh_rest[i] = 0.0f;
        }
    }
    
    // Helper to get base color (DC component only)
    Eigen::Vector3f GetBaseColor() const {
        return sh_dc;
    }
};

/**
 * @brief Gaussian model containing collection of Gaussians
 */
class GaussianModel {
public:
    GaussianModel() = default;
    ~GaussianModel() = default;
    
    // Load from PLY file
    bool LoadFromPLY(const std::string& ply_path);
    
    // Save to PLY file
    bool SaveToPLY(const std::string& ply_path) const;
    
    // Getters
    size_t GetNumGaussians() const { return m_gaussians.size(); }
    const std::vector<Gaussian>& GetGaussians() const { return m_gaussians; }
    std::vector<Gaussian>& GetGaussians() { return m_gaussians; }
    const Gaussian& GetGaussian(size_t index) const { return m_gaussians[index]; }
    
    // Clear all data
    void Clear() { m_gaussians.clear(); }
    
private:
    std::vector<Gaussian> m_gaussians;
};
