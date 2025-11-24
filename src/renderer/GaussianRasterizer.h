/**
 * @file      GaussianRasterizer.h
 * @brief     CPU Gaussian Splatting Rasterizer (based on original CUDA implementation)
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 * @reference Original CUDA implementation from 3D Gaussian Splatting (Kerbl et al. 2023)
 *            https://github.com/graphdeco-inria/diff-gaussian-rasterization
 */

#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "../database/GaussianModel.h"
#include "../database/Camera.h"

namespace gaussian_rasterizer {

/**
 * @brief Rasterization settings matching original implementation
 */
struct RasterSettings {
    int image_width;
    int image_height;
    float tan_fovx;
    float tan_fovy;
    float focal_x;
    float focal_y;
    Eigen::Vector3f bg_color;
    float scale_modifier;
    Eigen::Matrix4f viewmatrix;
    Eigen::Matrix4f projmatrix;
    Eigen::Vector3f campos;
    int sh_degree;
    float alpha_threshold;  // Minimum alpha value to render (default: 1/255)
    bool prefiltered;
    bool debug;
};

/**
 * @brief Rendered image output
 */
struct RenderOutput {
    std::vector<float> color;      // RGB image (H * W * 3)
    std::vector<float> depth;      // Depth image (H * W)
    std::vector<float> opacity;    // Opacity/alpha image (H * W)
    std::vector<int> n_contrib;    // Number of contributing Gaussians per pixel
};

/**
 * @brief CPU Gaussian Rasterizer - direct port from CUDA original
 */
class GaussianRasterizer {
public:
    GaussianRasterizer();
    ~GaussianRasterizer();
    
    /**
     * @brief Render Gaussians to image (main entry point)
     * @param gaussians Input Gaussian primitives
     * @param settings Rasterization settings
     * @return Rendered output
     */
    RenderOutput Render(const std::vector<Gaussian>& gaussians, const RasterSettings& settings);

private:
    // Forward pass functions (matching CUDA implementation)
    
    /**
     * @brief Compute RGB color from Spherical Harmonics
     * @param idx Gaussian index
     * @param deg SH degree (0-3)
     * @param means Gaussian positions
     * @param campos Camera position
     * @param shs SH coefficients [N, max_coeffs, 3]
     * @param result Output RGB color
     */
    void ComputeColorFromSH(
        int idx,
        int deg,
        const std::vector<Eigen::Vector3f>& means,
        const Eigen::Vector3f& campos,
        const std::vector<float>& shs,
        Eigen::Vector3f& result);
    
    /**
     * @brief Compute 3D covariance from scale and rotation
     * @param scale Scale vector
     * @param scale_modifier Scale multiplier
     * @param rot Rotation quaternion (w, x, y, z)
     * @param cov3D Output 3D covariance (6 elements: upper triangular)
     */
    void ComputeCov3D(
        const Eigen::Vector3f& scale,
        float scale_modifier,
        const Eigen::Quaternionf& rot,
        float* cov3D);
    
    /**
     * @brief Project 3D covariance to 2D screen space
     * @param mean 3D position
     * @param focal_x Focal length X
     * @param focal_y Focal length Y
     * @param tan_fovx Tangent of FOV X
     * @param tan_fovy Tangent of FOV Y
     * @param cov3D 3D covariance (6 elements)
     * @param viewmatrix View transformation matrix
     * @return 2D covariance (3 elements: cov_xx, cov_xy, cov_yy)
     */
    Eigen::Vector3f ComputeCov2D(
        const Eigen::Vector3f& mean,
        float focal_x,
        float focal_y,
        float tan_fovx,
        float tan_fovy,
        const float* cov3D,
        const Eigen::Matrix4f& viewmatrix);
    
    /**
     * @brief Preprocess Gaussians (projection, culling, sorting)
     */
    void Preprocess(
        const std::vector<Gaussian>& gaussians,
        const RasterSettings& settings,
        std::vector<int>& visible_indices,
        std::vector<float>& depths,
        std::vector<Eigen::Vector2f>& points_2d,
        std::vector<Eigen::Vector3f>& colors,
        std::vector<Eigen::Vector3f>& conics,  // inverse 2D covariance
        std::vector<float>& opacities,
        std::vector<float>& radii);
    
    /**
     * @brief Rasterize visible Gaussians to image (alpha blending)
     */
    void RasterizeToImage(
        const std::vector<int>& visible_indices,
        const std::vector<float>& depths,
        const std::vector<Eigen::Vector2f>& points_2d,
        const std::vector<Eigen::Vector3f>& colors,
        const std::vector<Eigen::Vector3f>& conics,
        const std::vector<float>& opacities,
        const std::vector<float>& radii,
        const RasterSettings& settings,
        RenderOutput& output);
    
    // Helper functions
    Eigen::Vector3f TransformPoint4x3(const Eigen::Vector3f& p, const Eigen::Matrix4f& matrix);
    Eigen::Vector4f TransformPoint4x4(const Eigen::Vector3f& p, const Eigen::Matrix4f& matrix);
    
    // SH constants (matching original)
    static constexpr float SH_C0 = 0.28209479177387814f;
    static constexpr float SH_C1 = 0.4886025119029199f;
    static constexpr float SH_C2[5] = {
        1.0925484305920792f,
        -1.0925484305920792f,
        0.31539156525252005f,
        -1.0925484305920792f,
        0.5462742152960396f
    };
    static constexpr float SH_C3[7] = {
        -0.5900435899266435f,
        2.890611442640554f,
        -0.4570457994644658f,
        0.3731763325901154f,
        -0.4570457994644658f,
        1.445305721320277f,
        -0.5900435899266435f
    };
};

} // namespace gaussian_rasterizer
