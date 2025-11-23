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
    Eigen::Vector3f color_dc;       // DC component of Spherical Harmonics (RGB)
    
    Gaussian()
        : position(Eigen::Vector3f::Zero())
        , scale(Eigen::Vector3f::Ones())
        , rotation(Eigen::Quaternionf::Identity())
        , opacity(1.0f)
        , color_dc(Eigen::Vector3f::Zero())
    {}
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
    
    // Clear all data
    void Clear() { m_gaussians.clear(); }
    
private:
    std::vector<Gaussian> m_gaussians;
};
