/**
 * @file      GaussianViewer.h
 * @brief     Pangolin-based viewer for Gaussian visualization
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 */

#pragma once

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <vector>
#include "../database/GaussianModel.h"

class GaussianViewer {
public:
    GaussianViewer();
    ~GaussianViewer();
    
    /**
     * @brief Initialize viewer window
     * @param width Window width
     * @param height Window height
     * @return true if successful
     */
    bool Initialize(int width = 1280, int height = 720);
    
    /**
     * @brief Update Gaussians to visualize
     * @param gaussians Vector of Gaussians
     */
    void UpdateGaussians(const std::vector<Gaussian>& gaussians);
    
    /**
     * @brief Main render loop (blocking)
     */
    void Run();
    
    /**
     * @brief Check if viewer window should close
     */
    bool ShouldClose() const;
    
    /**
     * @brief Shutdown viewer
     */
    void Shutdown();
    
private:
    void DrawGaussians();
    void DrawCoordinateFrame();
    void ComputeScaleRange();
    
    std::vector<Gaussian> m_gaussians;
    bool m_initialized;
    
    // Scale normalization
    float m_min_scale;
    float m_max_scale;
    
    // Pangolin camera state
    std::unique_ptr<pangolin::OpenGlRenderState> m_s_cam;
};
