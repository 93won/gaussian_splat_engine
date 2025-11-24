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
#include <opencv2/opencv.hpp>
#include <vector>
#include "../database/GaussianModel.h"
#include "../renderer/GaussianRasterizer.h"
#include "../database/Camera.h"

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
     * @brief Update Gaussian model (uploads to GPU for rendering)
     * @param model GaussianModel to upload
     */
    void UpdateGaussianModel(const GaussianModel& model);
    
    /**
     * @brief Update camera trajectory to visualize
     * @param poses Vector of 4x4 transformation matrices (world-to-camera)
     */
    void UpdateCameraTrajectory(const std::vector<Eigen::Matrix4d>& poses);
    
    /**
     * @brief Set camera intrinsics for frustum visualization
     * @param width Image width
     * @param height Image height
     * @param fx Focal length x
     * @param fy Focal length y
     */
    void SetCameraIntrinsics(int width, int height, float fx, float fy);
    
    /**
     * @brief Main render loop (blocking)
     */
    void Run();
    
    /**
     * @brief Render current camera view to image
     * @param camera_index Index of camera pose to render from
     * @param width Output image width
     * @param height Output image height
     * @return RGB image as cv::Mat
     */
    cv::Mat RenderCameraView(int camera_index, int width, int height);
    
    /**
     * @brief Render using Gaussian Splatting (CPU)
     * @param camera_index Index of camera pose to render from
     * @param width Output image width
     * @param height Output image height
     * @return RGB image as cv::Mat
     */
    cv::Mat RenderCPUGaussianSplatting(int camera_index, int width, int height, float scale_modifier = 1.0f);
    
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
    void DrawCameraTrajectory();
    void DrawCameraFrustum(const Eigen::Matrix4d& T_wc);
    void ComputeScaleRange();
    void ProjectGaussianToImage(const Gaussian& g, const Eigen::Matrix4d& T_cw, 
                                cv::Mat& image, int width, int height);
    
    std::vector<Gaussian> m_gaussians;
    // Keep a copy of the model for CPU rendering
    GaussianModel m_model;

    std::vector<Eigen::Matrix4d> m_camera_poses;
    bool m_initialized;
    
    // Current camera index for rendering
    int m_current_camera_index;
    
    // Gaussian renderer
    std::unique_ptr<gaussian_rasterizer::GaussianRasterizer> m_rasterizer;
    
    // Camera intrinsics for frustum drawing
    int m_cam_width;
    int m_cam_height;
    float m_cam_fx;
    float m_cam_fy;
    
    // Scale normalization
    float m_min_scale;
    float m_max_scale;
    
    // Pangolin camera state
    std::unique_ptr<pangolin::OpenGlRenderState> m_s_cam;
};
