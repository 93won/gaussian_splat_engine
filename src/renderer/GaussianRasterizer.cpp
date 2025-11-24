/**
 * @file      GaussianRasterizer.cpp
 * @brief     CPU Gaussian Splatting Rasterizer implementation
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 * 
 * @license   Research use only (non-commercial)
 * 
 * This implementation follows the algorithm described in:
 * "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
 * Kerbl et al., ACM TOG 2023
 * 
 * Implementation details referenced from the official open-source code:
 * graphdeco-inria/diff-gaussian-rasterization
 * 
 * For research and educational purposes only.
 */

#include "GaussianRasterizer.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace gaussian_rasterizer {

GaussianRasterizer::GaussianRasterizer() {
}

GaussianRasterizer::~GaussianRasterizer() {
}

Eigen::Vector3f GaussianRasterizer::TransformPoint4x3(const Eigen::Vector3f& p, const Eigen::Matrix4f& matrix) {
    Eigen::Vector4f p_hom(p.x(), p.y(), p.z(), 1.0f);
    Eigen::Vector4f transformed = matrix * p_hom;
    return Eigen::Vector3f(transformed.x(), transformed.y(), transformed.z());
}

Eigen::Vector4f GaussianRasterizer::TransformPoint4x4(const Eigen::Vector3f& p, const Eigen::Matrix4f& matrix) {
    Eigen::Vector4f p_hom(p.x(), p.y(), p.z(), 1.0f);
    return matrix * p_hom;
}

void GaussianRasterizer::ComputeColorFromSH(
    int idx,
    int deg,
    const std::vector<Eigen::Vector3f>& means,
    const Eigen::Vector3f& campos,
    const std::vector<float>& shs,
    Eigen::Vector3f& result)
{
    // Based on "Differentiable Point-Based Radiance Fields" (Zhang et al. 2022)
    // and original 3DGS CUDA implementation
    
    Eigen::Vector3f pos = means[idx];
    Eigen::Vector3f dir = pos - campos;
    dir.normalize();
    
    // SH coefficients for this Gaussian: [max_coeffs, 3] layout
    // We expect shs to be organized as: [N * max_coeffs * 3]
    int max_coeffs = (deg + 1) * (deg + 1);
    const float* sh = &shs[idx * max_coeffs * 3];
    
    // Degree 0 (DC term) - sh organized as [coeff0_rgb, coeff1_rgb, ...]
    result = Eigen::Vector3f(
        SH_C0 * sh[0],
        SH_C0 * sh[1],
        SH_C0 * sh[2]
    );
    
    if (deg > 0) {
        float x = dir.x();
        float y = dir.y();
        float z = dir.z();
        
        // Degree 1 (3 coefficients)
        result.x() += -SH_C1 * y * sh[3] + SH_C1 * z * sh[6] - SH_C1 * x * sh[9];
        result.y() += -SH_C1 * y * sh[4] + SH_C1 * z * sh[7] - SH_C1 * x * sh[10];
        result.z() += -SH_C1 * y * sh[5] + SH_C1 * z * sh[8] - SH_C1 * x * sh[11];
        
        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            
            // Degree 2 (5 coefficients)
            result.x() += SH_C2[0] * xy * sh[12] +
                         SH_C2[1] * yz * sh[15] +
                         SH_C2[2] * (2.0f * zz - xx - yy) * sh[18] +
                         SH_C2[3] * xz * sh[21] +
                         SH_C2[4] * (xx - yy) * sh[24];
            
            result.y() += SH_C2[0] * xy * sh[13] +
                         SH_C2[1] * yz * sh[16] +
                         SH_C2[2] * (2.0f * zz - xx - yy) * sh[19] +
                         SH_C2[3] * xz * sh[22] +
                         SH_C2[4] * (xx - yy) * sh[25];
            
            result.z() += SH_C2[0] * xy * sh[14] +
                         SH_C2[1] * yz * sh[17] +
                         SH_C2[2] * (2.0f * zz - xx - yy) * sh[20] +
                         SH_C2[3] * xz * sh[23] +
                         SH_C2[4] * (xx - yy) * sh[26];
            
            if (deg > 2) {
                // Degree 3 (7 coefficients)
                result.x() += SH_C3[0] * y * (3.0f * xx - yy) * sh[27] +
                             SH_C3[1] * xy * z * sh[30] +
                             SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[33] +
                             SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[36] +
                             SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[39] +
                             SH_C3[5] * z * (xx - yy) * sh[42] +
                             SH_C3[6] * x * (xx - 3.0f * yy) * sh[45];
                
                result.y() += SH_C3[0] * y * (3.0f * xx - yy) * sh[28] +
                             SH_C3[1] * xy * z * sh[31] +
                             SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[34] +
                             SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[37] +
                             SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[40] +
                             SH_C3[5] * z * (xx - yy) * sh[43] +
                             SH_C3[6] * x * (xx - 3.0f * yy) * sh[46];
                
                result.z() += SH_C3[0] * y * (3.0f * xx - yy) * sh[29] +
                             SH_C3[1] * xy * z * sh[32] +
                             SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[35] +
                             SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[38] +
                             SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[41] +
                             SH_C3[5] * z * (xx - yy) * sh[44] +
                             SH_C3[6] * x * (xx - 3.0f * yy) * sh[47];
            }
        }
    }
    
    // Add 0.5 offset (matching original)
    result += Eigen::Vector3f(0.5f, 0.5f, 0.5f);
    
    // Clamp to non-negative
    result.x() = std::max(0.0f, result.x());
    result.y() = std::max(0.0f, result.y());
    result.z() = std::max(0.0f, result.z());
}

void GaussianRasterizer::ComputeCov3D(
    const Eigen::Vector3f& scale,
    float scale_modifier,
    const Eigen::Quaternionf& rot,
    float* cov3D)
{
    // Create scaling matrix
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    S(0, 0) = scale_modifier * scale.x();
    S(1, 1) = scale_modifier * scale.y();
    S(2, 2) = scale_modifier * scale.z();
    
    // Normalize quaternion and extract components
    Eigen::Quaternionf q = rot.normalized();
    float r = q.w();  // Real part
    float x = q.x();
    float y = q.y();
    float z = q.z();
    
    // Compute rotation matrix from quaternion
    Eigen::Matrix3f R;
    R(0, 0) = 1.0f - 2.0f * (y * y + z * z);
    R(0, 1) = 2.0f * (x * y - r * z);
    R(0, 2) = 2.0f * (x * z + r * y);
    R(1, 0) = 2.0f * (x * y + r * z);
    R(1, 1) = 1.0f - 2.0f * (x * x + z * z);
    R(1, 2) = 2.0f * (y * z - r * x);
    R(2, 0) = 2.0f * (x * z - r * y);
    R(2, 1) = 2.0f * (y * z + r * x);
    R(2, 2) = 1.0f - 2.0f * (x * x + y * y);
    
    // M = S * R
    Eigen::Matrix3f M = S * R;
    
    // Compute 3D covariance: Sigma = M^T * M
    Eigen::Matrix3f Sigma = M.transpose() * M;
    
    // Store upper triangular part (symmetric matrix)
    cov3D[0] = Sigma(0, 0);
    cov3D[1] = Sigma(0, 1);
    cov3D[2] = Sigma(0, 2);
    cov3D[3] = Sigma(1, 1);
    cov3D[4] = Sigma(1, 2);
    cov3D[5] = Sigma(2, 2);
}

Eigen::Vector3f GaussianRasterizer::ComputeCov2D(
    const Eigen::Vector3f& mean,
    float focal_x,
    float focal_y,
    float tan_fovx,
    float tan_fovy,
    const float* cov3D,
    const Eigen::Matrix4f& viewmatrix)
{
    // Transform point to camera space
    Eigen::Vector3f t = TransformPoint4x3(mean, viewmatrix);
    
    // Clamp to FOV (avoid extreme projections)
    float limx = 1.3f * tan_fovx;
    float limy = 1.3f * tan_fovy;
    float txtz = t.x() / t.z();
    float tytz = t.y() / t.z();
    t.x() = std::min(limx, std::max(-limx, txtz)) * t.z();
    t.y() = std::min(limy, std::max(-limy, tytz)) * t.z();
    
    // Jacobian of perspective projection
    Eigen::Matrix3f J;
    J << focal_x / t.z(), 0.0f, -(focal_x * t.x()) / (t.z() * t.z()),
         0.0f, focal_y / t.z(), -(focal_y * t.y()) / (t.z() * t.z()),
         0.0f, 0.0f, 0.0f;
    
    // Extract rotation part of view matrix (upper-left 3x3)
    Eigen::Matrix3f W;
    W << viewmatrix(0, 0), viewmatrix(0, 1), viewmatrix(0, 2),
         viewmatrix(1, 0), viewmatrix(1, 1), viewmatrix(1, 2),
         viewmatrix(2, 0), viewmatrix(2, 1), viewmatrix(2, 2);
    
    // T = W * J
    Eigen::Matrix3f T = W * J;
    
    // Reconstruct 3D covariance matrix (symmetric)
    Eigen::Matrix3f Vrk;
    Vrk << cov3D[0], cov3D[1], cov3D[2],
           cov3D[1], cov3D[3], cov3D[4],
           cov3D[2], cov3D[4], cov3D[5];
    
    // Project to 2D: cov2D = T^T * Vrk * T
    Eigen::Matrix3f cov = T.transpose() * Vrk * T;
    
    // Apply low-pass filter (minimum 0.3 pixels)
    cov(0, 0) += 0.3f;
    cov(1, 1) += 0.3f;
    
    // Return upper triangular (symmetric)
    return Eigen::Vector3f(cov(0, 0), cov(0, 1), cov(1, 1));
}

void GaussianRasterizer::Preprocess(
    const std::vector<Gaussian>& gaussians,
    const RasterSettings& settings,
    std::vector<int>& visible_indices,
    std::vector<float>& depths,
    std::vector<Eigen::Vector2f>& points_2d,
    std::vector<Eigen::Vector3f>& colors,
    std::vector<Eigen::Vector3f>& conics,
    std::vector<float>& opacities,
    std::vector<float>& radii)
{
    visible_indices.clear();
    depths.clear();
    points_2d.clear();
    colors.clear();
    conics.clear();
    opacities.clear();
    
    int W = settings.image_width;
    int H = settings.image_height;
    
    // Prepare SH coefficients in [N, max_coeffs, 3] format
    int max_coeffs = (settings.sh_degree + 1) * (settings.sh_degree + 1);
    std::vector<float> shs_all;
    std::vector<Eigen::Vector3f> means;
    
    for (const auto& g : gaussians) {
        means.push_back(g.position);
        
        // Pack SH coefficients: DC + rest in [max_coeffs, 3] layout
        // Original format: sh_dc[3], sh_rest[45] where rest is [R_0..R_14, G_0..G_14, B_0..B_14]
        // Target format: [coeff0_R, coeff0_G, coeff0_B, coeff1_R, coeff1_G, coeff1_B, ...]
        
        // DC term
        shs_all.push_back(g.sh_dc.x());
        shs_all.push_back(g.sh_dc.y());
        shs_all.push_back(g.sh_dc.z());
        
        // Rest terms: need to transpose from [R_all, G_all, B_all] to [coeff_RGB, ...]
        int num_rest = max_coeffs - 1;  // Exclude DC
        for (int coeff = 0; coeff < num_rest; ++coeff) {
            shs_all.push_back(g.sh_rest[coeff]);           // R
            shs_all.push_back(g.sh_rest[coeff + 15]);      // G
            shs_all.push_back(g.sh_rest[coeff + 30]);      // B
        }
    }
    
    // Process each Gaussian
    for (size_t idx = 0; idx < gaussians.size(); ++idx) {
        const auto& g = gaussians[idx];
        
        // Transform to clip space
        Eigen::Vector4f p_hom = TransformPoint4x4(g.position, settings.projmatrix);
        float p_w = 1.0f / (p_hom.w() + 0.0000001f);
        Eigen::Vector3f p_proj(p_hom.x() * p_w, p_hom.y() * p_w, p_hom.z() * p_w);
        
        // Transform to camera space for depth
        Eigen::Vector3f p_view = TransformPoint4x3(g.position, settings.viewmatrix);
        float depth = p_view.z();
        
        // Frustum culling
        if (depth <= 0.0f) continue;
        if (p_proj.x() < -1.3f || p_proj.x() > 1.3f) continue;
        if (p_proj.y() < -1.3f || p_proj.y() > 1.3f) continue;
        
        // Compute 3D covariance
        float cov3D[6];
        // Apply exp activation to scale
        Eigen::Vector3f scale_act(std::exp(g.scale.x()), 
                                   std::exp(g.scale.y()), 
                                   std::exp(g.scale.z()));
        ComputeCov3D(scale_act, settings.scale_modifier, g.rotation, cov3D);
        
        // Project to 2D
        Eigen::Vector3f cov2D = ComputeCov2D(
            g.position, settings.focal_x, settings.focal_y,
            settings.tan_fovx, settings.tan_fovy,
            cov3D, settings.viewmatrix);
        
        // Compute determinant (matching original 3DGS implementation)
        float det = cov2D.x() * cov2D.z() - cov2D.y() * cov2D.y();
        if (det == 0.0f) continue;
        
        // Compute inverse (conic)
        float det_inv = 1.0f / det;
        Eigen::Vector3f conic(cov2D.z() * det_inv, -cov2D.y() * det_inv, cov2D.x() * det_inv);
        
        // Compute radius (3 sigma rule - matching original CUDA implementation)
        float mid = 0.5f * (cov2D.x() + cov2D.z());
        float lambda1 = mid + std::sqrt(std::max(0.1f, mid * mid - det));
        float lambda2 = mid - std::sqrt(std::max(0.1f, mid * mid - det));
        float my_radius = 3.0f * std::sqrt(std::max(lambda1, lambda2));
        float radius = std::ceil(my_radius);
        
        // Convert NDC to pixel coordinates
        Eigen::Vector2f point_2d(
            (p_proj.x() * 0.5f + 0.5f) * W,
            (p_proj.y() * 0.5f + 0.5f) * H
        );
        
        // Check if within image bounds (with radius)
        if (point_2d.x() + radius < 0 || point_2d.x() - radius >= W) continue;
        if (point_2d.y() + radius < 0 || point_2d.y() - radius >= H) continue;
        
        // Compute color from SH
        Eigen::Vector3f color;
        ComputeColorFromSH(idx, settings.sh_degree, means, settings.campos, shs_all, color);
        
        // Apply sigmoid to opacity
        float opacity = 1.0f / (1.0f + std::exp(-g.opacity));
        
        // Store visible Gaussian
        visible_indices.push_back(idx);
        depths.push_back(depth);
        points_2d.push_back(point_2d);
        colors.push_back(color);
        conics.push_back(conic);
        opacities.push_back(opacity);
        radii.push_back(radius);
    }
    
    // Sort by depth (front to back)
    std::vector<size_t> sort_indices(visible_indices.size());
    for (size_t i = 0; i < sort_indices.size(); ++i) {
        sort_indices[i] = i;
    }
    
    std::sort(sort_indices.begin(), sort_indices.end(),
        [&depths](size_t a, size_t b) { return depths[a] < depths[b]; });
    
    // Reorder all arrays
    std::vector<int> sorted_visible_indices;
    std::vector<float> sorted_depths;
    std::vector<Eigen::Vector2f> sorted_points_2d;
    std::vector<Eigen::Vector3f> sorted_colors;
    std::vector<Eigen::Vector3f> sorted_conics;
    std::vector<float> sorted_opacities;
    std::vector<float> sorted_radii;
    
    for (size_t idx : sort_indices) {
        sorted_visible_indices.push_back(visible_indices[idx]);
        sorted_depths.push_back(depths[idx]);
        sorted_points_2d.push_back(points_2d[idx]);
        sorted_colors.push_back(colors[idx]);
        sorted_conics.push_back(conics[idx]);
        sorted_opacities.push_back(opacities[idx]);
        sorted_radii.push_back(radii[idx]);
    }
    
    visible_indices = sorted_visible_indices;
    depths = sorted_depths;
    points_2d = sorted_points_2d;
    colors = sorted_colors;
    conics = sorted_conics;
    opacities = sorted_opacities;
    radii = sorted_radii;
}

void GaussianRasterizer::RasterizeToImage(
    const std::vector<int>& visible_indices,
    const std::vector<float>& depths,
    const std::vector<Eigen::Vector2f>& points_2d,
    const std::vector<Eigen::Vector3f>& colors,
    const std::vector<Eigen::Vector3f>& conics,
    const std::vector<float>& opacities,
    const std::vector<float>& radii,
    const RasterSettings& settings,
    RenderOutput& output)
{
    int W = settings.image_width;
    int H = settings.image_height;
    
    // Initialize output
    output.color.resize(H * W * 3, 0.0f);
    output.depth.resize(H * W, 0.0f);
    output.opacity.resize(H * W, 0.0f);
    output.n_contrib.resize(H * W, 0);
    
    // Initialize per-pixel transmittance buffer
    std::vector<float> T(H * W, 1.0f);
    
    std::cout << "[RasterizeToImage] Processing " << visible_indices.size() << " Gaussians..." << std::endl;
    
    // Iterate over sorted Gaussians (front-to-back)
    for (size_t i = 0; i < visible_indices.size(); ++i) {
        Eigen::Vector2f xy = points_2d[i];
        float radius = radii[i];
        Eigen::Vector3f color = colors[i];
        Eigen::Vector3f con_o = conics[i];
        float opacity = opacities[i];
        float depth = depths[i];
        
        // Compute bounding box
        int min_x = std::max(0, (int)(xy.x() - radius));
        int max_x = std::min(W - 1, (int)(xy.x() + radius));
        int min_y = std::max(0, (int)(xy.y() - radius));
        int max_y = std::min(H - 1, (int)(xy.y() + radius));
        
        // Render to pixels within bounding box
        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                int pix_id = y * W + x;
                
                // Check transmittance - skip if already saturated
                if (T[pix_id] < 0.0001f) continue;
                
                Eigen::Vector2f pixf(x, y);
                Eigen::Vector2f d = xy - pixf;
                
                // Evaluate 2D Gaussian
                float power = -0.5f * (con_o.x() * d.x() * d.x() + con_o.z() * d.y() * d.y()) 
                             - con_o.y() * d.x() * d.y();
                
                if (power > 0.0f) continue;
                
                // Compute alpha and filter very transparent contributions
                float alpha = std::min(0.99f, opacity * std::exp(power));
                if (alpha < settings.alpha_threshold) continue;  // User-configurable threshold
                
                // Alpha blending
                float test_T = T[pix_id] * (1.0f - alpha);
                output.color[pix_id * 3 + 0] += color.x() * alpha * T[pix_id];
                output.color[pix_id * 3 + 1] += color.y() * alpha * T[pix_id];
                output.color[pix_id * 3 + 2] += color.z() * alpha * T[pix_id];
                output.depth[pix_id] += depth * alpha * T[pix_id];
                T[pix_id] = test_T;
                output.n_contrib[pix_id]++;
            }
        }
    }
    
    // Add background color where T > 0
    #pragma omp parallel for
    for (int pix_id = 0; pix_id < W * H; ++pix_id) {
        output.color[pix_id * 3 + 0] += T[pix_id] * settings.bg_color.x();
        output.color[pix_id * 3 + 1] += T[pix_id] * settings.bg_color.y();
        output.color[pix_id * 3 + 2] += T[pix_id] * settings.bg_color.z();
        output.opacity[pix_id] = 1.0f - T[pix_id];
    }
    
    // Debug: Print sample pixel values
    if (settings.debug) {
        std::cout << "[RasterizeToImage] First 5 pixels:" << std::endl;
        for (int i = 0; i < 5 && i < W * H; ++i) {
            std::cout << "  Pixel[" << i << "]: RGB=(" 
                      << output.color[i*3+0] << ", " 
                      << output.color[i*3+1] << ", " 
                      << output.color[i*3+2] << ")" 
                      << " opacity=" << output.opacity[i]
                      << " contrib=" << output.n_contrib[i] << std::endl;
        }
    }
}

RenderOutput GaussianRasterizer::Render(
    const std::vector<Gaussian>& gaussians,
    const RasterSettings& settings)
{
    std::cout << "[GaussianRasterizer::Render] Called with " << gaussians.size() << " Gaussians" << std::endl;
    
    RenderOutput output;
    
    if (gaussians.empty()) {
        std::cerr << "[GaussianRasterizer] No Gaussians to render" << std::endl;
        return output;
    }
    
    std::cout << "[GaussianRasterizer::Render] Starting preprocessing..." << std::endl;
    
    // Step 1: Preprocess (projection, culling, sorting)
    std::vector<int> visible_indices;
    std::vector<float> depths;
    std::vector<Eigen::Vector2f> points_2d;
    std::vector<Eigen::Vector3f> colors;
    std::vector<Eigen::Vector3f> conics;
    std::vector<float> opacities;
    std::vector<float> radii;
    
    Preprocess(gaussians, settings, visible_indices, depths, 
               points_2d, colors, conics, opacities, radii);
    
    if (settings.debug) {
        std::cout << "[GaussianRasterizer] Total Gaussians: " << gaussians.size() << std::endl;
        std::cout << "[GaussianRasterizer] Visible Gaussians: " << visible_indices.size() << std::endl;
    }
    
    std::cout << "[GaussianRasterizer::Render] Starting rasterization..." << std::endl;
    
    // Step 2: Rasterize to image
    RasterizeToImage(visible_indices, depths, points_2d, colors, conics, opacities, radii, settings, output);
    
    std::cout << "[GaussianRasterizer::Render] Rasterization complete!" << std::endl;
    
    return output;
}

} // namespace gaussian_rasterizer
