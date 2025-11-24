/**
 * @file      GaussianViewer.cpp
 * @brief     Pangolin-based viewer for Gaussian visualization
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 */

#include "GaussianViewer.h"
#include <iostream>
#include <cmath>
#include <limits>
#include <memory>

GaussianViewer::GaussianViewer() 
    : m_initialized(false)
    , m_current_camera_index(0)
    , m_cam_width(640)
    , m_cam_height(480)
    , m_cam_fx(500.0f)
    , m_cam_fy(500.0f)
    , m_min_scale(0.0f)
    , m_max_scale(1.0f) 
    , m_s_cam(nullptr) {
}

GaussianViewer::~GaussianViewer() {
    Shutdown();
}

bool GaussianViewer::Initialize(int width, int height) {
    if (m_initialized) {
        std::cerr << "[GaussianViewer] Already initialized" << std::endl;
        return false;
    }
    
    // Create Pangolin window
    pangolin::CreateWindowAndBind("Gaussian Splatting Viewer", width, height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Set black background
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Define Projection and initial ModelView matrix (store as member)
    m_s_cam = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(width, height, 500, 500, width/2, height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -2, -5, 0, 0, 0, pangolin::AxisY)
    );
    
    // Create Interactive View in window with name "cam"
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(250), 1.0, -(float)width/height)
        .SetHandler(new pangolin::Handler3D(*m_s_cam));
    
    // Initialize CPU renderer
    m_cpu_renderer = std::make_unique<gaussian_renderer::CPUGaussianRenderer>();
    std::cout << "[GaussianViewer] CPU renderer created" << std::endl;
    
    m_initialized = true;
    
    std::cout << "[GaussianViewer] Initialized (" << width << "x" << height << ")" << std::endl;
    return true;
}

void GaussianViewer::UpdateGaussians(const std::vector<Gaussian>& gaussians) {
    m_gaussians = gaussians;
    m_model.GetGaussians() = gaussians; // Sync model
    ComputeScaleRange();
    std::cout << "[GaussianViewer] Updated " << m_gaussians.size() << " Gaussians" << std::endl;
    std::cout << "[GaussianViewer] Scale range: [" << m_min_scale << ", " << m_max_scale << "]" << std::endl;
}

void GaussianViewer::UpdateGaussianModel(const GaussianModel& model) {
    m_model = model; // Keep copy for CPU
    m_gaussians = model.GetGaussians(); // Sync vector
    ComputeScaleRange();
}

void GaussianViewer::ComputeScaleRange() {
    if (m_gaussians.empty()) {
        m_min_scale = 0.0f;
        m_max_scale = 1.0f;
        return;
    }
    
    m_min_scale = std::numeric_limits<float>::max();
    m_max_scale = std::numeric_limits<float>::lowest();
    
    for (const auto& g : m_gaussians) {
        // Get max scale component for each Gaussian
        float max_component = std::max({g.scale.x(), g.scale.y(), g.scale.z()});
        m_min_scale = std::min(m_min_scale, max_component);
        m_max_scale = std::max(m_max_scale, max_component);
    }
}

void GaussianViewer::DrawCoordinateFrame() {
    const float axis_length = 0.5f;
    
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(axis_length, 0.0f, 0.0f);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, axis_length, 0.0f);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, axis_length);
    
    glEnd();
}

void GaussianViewer::DrawGaussians() {
    if (m_gaussians.empty()) {
        return;
    }
    
    // Don't use glBegin/glEnd with point size changes
    // Instead, draw each Gaussian individually with its own size
    for (const auto& g : m_gaussians) {
        // Sigmoid activation for opacity (inverse of logit)
        float opacity = 1.0f / (1.0f + std::exp(-g.opacity));
        
        // Filter out very transparent points to match rasterizer behavior
        if (opacity < 0.05f) continue;
        
        opacity = std::max(0.0f, std::min(1.0f, opacity));
        
        // Color DC (SH coefficient 0) to RGB
        const float SH_C0 = 0.28209479177387814f;
        float r = std::max(0.0f, g.sh_dc.x() * SH_C0 + 0.5f);
        float g_val = std::max(0.0f, g.sh_dc.y() * SH_C0 + 0.5f);
        float b = std::max(0.0f, g.sh_dc.z() * SH_C0 + 0.5f);
        
        // Calculate point size based on Gaussian's scale
        // Use exponential of scale (since scale is log-scale in Gaussian Splatting)
        float scale_x = std::exp(g.scale.x());
        float scale_y = std::exp(g.scale.y());
        float scale_z = std::exp(g.scale.z());
        float avg_scale = (scale_x + scale_y + scale_z) / 3.0f;
        
        // Don't set point size here - it's controlled globally
        glColor4f(r, g_val, b, opacity);
        glBegin(GL_POINTS);
        glVertex3f(g.position.x(), g.position.y(), g.position.z());
        glEnd();
    }
}

void GaussianViewer::Run() {
    if (!m_initialized) {
        std::cerr << "[GaussianViewer] Not initialized!" << std::endl;
        return;
    }
    
    if (!m_cpu_renderer) {
        std::cerr << "[GaussianViewer] CPU renderer not initialized!" << std::endl;
        return;
    }
    
    // Create display layout
    // Left side: UI panel (400px wide) with buttons at top and two images at bottom
    // Right side: 3D view (remaining space)
    
    int panel_width = 400;
    int ui_height = 250;  // Height for UI controls
    int image_height = 480 / 2;  // Half height for each image
    
    // UI panel at top-left
    pangolin::CreatePanel("ui")
        .SetBounds(pangolin::Attach::Pix(2*image_height), 1.0, 0.0, pangolin::Attach::Pix(panel_width));
    
    // Point rendering image at middle-left
    pangolin::View& d_image_points = pangolin::Display("image_points")
        .SetBounds(pangolin::Attach::Pix(image_height), pangolin::Attach::Pix(2*image_height), 
                   0.0, pangolin::Attach::Pix(panel_width))
        .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    
    // Gaussian splatting image at bottom-left
    pangolin::View& d_image_splat = pangolin::Display("image_splat")
        .SetBounds(0.0, pangolin::Attach::Pix(image_height), 
                   0.0, pangolin::Attach::Pix(panel_width))
        .SetLock(pangolin::LockLeft, pangolin::LockBottom);
    
    // 3D view on the right side (full height)
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0, -640.0f/480.0f)
        .SetHandler(new pangolin::Handler3D(*m_s_cam));
    
    // Create textures for rendered images
    pangolin::GlTexture imageTexture1(m_cam_width, m_cam_height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    pangolin::GlTexture imageTexture2(m_cam_width, m_cam_height, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
    
    // UI variables
    pangolin::Var<float> gaussian_scale("ui.Gaussian Scale", 1.0f, 0.001f, 1.0f);
    pangolin::Var<bool> show_coordinate_frame("ui.Show Axes", true, true);
    pangolin::Var<bool> show_camera_trajectory("ui.Show Camera Trajectory", true, true);
    pangolin::Var<float> point_size("ui.Point Size", 3.0f, 1.0f, 10.0f);
    pangolin::Var<bool> apply_sigmoid("ui.Apply Sigmoid to Color", false, true);
    pangolin::Var<bool> btn_next("ui.Next Camera", false, false);
    pangolin::Var<bool> btn_prev("ui.Prev Camera", false, false);
    pangolin::Var<int> camera_index("ui.Camera Index", 0, 0, std::max(0, (int)m_camera_poses.size()-1));
    
    std::cout << "[GaussianViewer] Starting render loop..." << std::endl;
    std::cout << "[GaussianViewer] Controls:" << std::endl;
    std::cout << "  - Drag to rotate 3D view" << std::endl;
    std::cout << "  - Right-click drag to pan" << std::endl;
    std::cout << "  - Scroll to zoom" << std::endl;
    std::cout << "  - Click 'Next Camera'/'Prev Camera' to switch views" << std::endl;
    std::cout << "  - Top image: Point rendering" << std::endl;
    std::cout << "  - Bottom image: Gaussian Splatting (CPU)" << std::endl;
    std::cout << "[GaussianViewer] Close window to exit" << std::endl;
    
    cv::Mat rendered_points, rendered_splat;
    bool need_rerender = true;
    float last_gaussian_scale = gaussian_scale;
    
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Handle camera switching
        if (pangolin::Pushed(btn_next)) {
            m_current_camera_index = (m_current_camera_index + 1) % m_camera_poses.size();
            camera_index = m_current_camera_index;
            need_rerender = true;
        }
        if (pangolin::Pushed(btn_prev)) {
            m_current_camera_index = (m_current_camera_index - 1 + m_camera_poses.size()) % m_camera_poses.size();
            camera_index = m_current_camera_index;
            need_rerender = true;
        }
        
        // Check if gaussian_scale changed
        if (std::abs(last_gaussian_scale - gaussian_scale) > 0.001f) {
            last_gaussian_scale = gaussian_scale;
            need_rerender = true;
        }
        
        // Render camera views if needed
        if (!m_camera_poses.empty() && need_rerender) {
            // Point rendering
            rendered_points = RenderCameraView(m_current_camera_index, m_cam_width, m_cam_height);
            if (!rendered_points.empty()) {
                imageTexture1.Upload(rendered_points.data, GL_BGR, GL_UNSIGNED_BYTE);
            }
            
            // Gaussian splatting rendering (CPU only)
            rendered_splat = RenderCPUGaussianSplatting(m_current_camera_index, m_cam_width, m_cam_height, gaussian_scale);

            if (!rendered_splat.empty()) {
                imageTexture2.Upload(rendered_splat.data, GL_BGR, GL_UNSIGNED_BYTE);
            }
            
            need_rerender = false;
        }
        
        // Draw 3D view
        d_cam.Activate(*m_s_cam);
        
        if (show_coordinate_frame) {
            DrawCoordinateFrame();
        }
        
        if (show_camera_trajectory) {
            DrawCameraTrajectory();
        }
        
        glEnable(GL_POINT_SPRITE);
        glEnable(GL_PROGRAM_POINT_SIZE);
        
        DrawGaussians();
        
        glDisable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_POINT_SPRITE);
        
        // Draw point rendering image (middle of left panel)
        if (!rendered_points.empty()) {
            d_image_points.Activate();
            glColor3f(1.0f, 1.0f, 1.0f);
            imageTexture1.RenderToViewportFlipY();
        }
        
        // Draw Gaussian splatting image (bottom of left panel)
        if (!rendered_splat.empty()) {
            d_image_splat.Activate();
            glColor3f(1.0f, 1.0f, 1.0f);
            imageTexture2.RenderToViewportFlipY();
        }
        
        pangolin::FinishFrame();
    }
}

bool GaussianViewer::ShouldClose() const {
    return pangolin::ShouldQuit();
}

void GaussianViewer::Shutdown() {
    if (m_initialized) {
        pangolin::DestroyWindow("Gaussian Splatting Viewer");
        m_initialized = false;
        std::cout << "[GaussianViewer] Shutdown" << std::endl;
    }
}

void GaussianViewer::UpdateCameraTrajectory(const std::vector<Eigen::Matrix4d>& poses) {
    m_camera_poses = poses;
    std::cout << "[GaussianViewer] Loaded " << poses.size() << " camera poses" << std::endl;
}

void GaussianViewer::SetCameraIntrinsics(int width, int height, float fx, float fy) {
    m_cam_width = width;
    m_cam_height = height;
    m_cam_fx = fx;
    m_cam_fy = fy;
    std::cout << "[GaussianViewer] Camera intrinsics set: " 
              << width << "x" << height 
              << " (fx=" << fx << ", fy=" << fy << ")" << std::endl;
}

void GaussianViewer::DrawCameraTrajectory() {
    if (m_camera_poses.empty()) return;
    
    // Draw camera trajectory line
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.0f, 0.0f); // Red line
    glBegin(GL_LINE_STRIP);
    for (const auto& T : m_camera_poses) {
        // Assuming T is camera-to-world (T_cw), camera center is just translation
        Eigen::Vector3d camera_center = T.block<3,1>(0,3);
        glVertex3d(camera_center.x(), camera_center.y(), camera_center.z());
    }
    glEnd();
    
    // Draw each camera frustum
    for (const auto& T : m_camera_poses) {
        DrawCameraFrustum(T);
    }
}

void GaussianViewer::DrawCameraFrustum(const Eigen::Matrix4d& T_cw) {
    // T_cw is camera-to-world transform
    Eigen::Matrix3d R = T_cw.block<3,3>(0,0);
    Eigen::Vector3d t = T_cw.block<3,1>(0,3);
    Eigen::Vector3d camera_center = t;  // Translation is camera position in world frame
    
    // Frustum size (depth from camera)
    float frustum_depth = 0.2f;
    
    // Image plane corners in camera frame (normalized coordinates)
    float half_width = (m_cam_width / 2.0f) / m_cam_fx * frustum_depth;
    float half_height = (m_cam_height / 2.0f) / m_cam_fy * frustum_depth;
    
    // 4 corners of frustum in camera frame
    Eigen::Vector3d corners_cam[4];
    corners_cam[0] = Eigen::Vector3d(-half_width, -half_height, frustum_depth); // Top-left
    corners_cam[1] = Eigen::Vector3d(half_width, -half_height, frustum_depth);  // Top-right
    corners_cam[2] = Eigen::Vector3d(half_width, half_height, frustum_depth);   // Bottom-right
    corners_cam[3] = Eigen::Vector3d(-half_width, half_height, frustum_depth);  // Bottom-left
    
    // Transform to world frame: p_world = R * p_cam + t
    Eigen::Vector3d corners_world[4];
    for (int i = 0; i < 4; i++) {
        corners_world[i] = R * corners_cam[i] + t;
    }
    
    // Draw frustum edges (from camera center to corners)
    glLineWidth(1.0f);
    glColor3f(0.0f, 1.0f, 0.0f); // Green frustum
    glBegin(GL_LINES);
    for (int i = 0; i < 4; i++) {
        glVertex3d(camera_center.x(), camera_center.y(), camera_center.z());
        glVertex3d(corners_world[i].x(), corners_world[i].y(), corners_world[i].z());
    }
    glEnd();
    
    // Draw image plane rectangle
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 4; i++) {
        glVertex3d(corners_world[i].x(), corners_world[i].y(), corners_world[i].z());
    }
    glEnd();
    
    // Draw camera center as small sphere
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow camera center
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    glVertex3d(camera_center.x(), camera_center.y(), camera_center.z());
    glEnd();
}

cv::Mat GaussianViewer::RenderCameraView(int camera_index, int width, int height) {
    if (camera_index < 0 || camera_index >= m_camera_poses.size()) {
        return cv::Mat();
    }
    
    std::cout << "[GaussianViewer] Rendering camera view " << camera_index << std::endl;
    
    // Create output image (black background)
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    
    // Get camera pose (camera-to-world)
    const Eigen::Matrix4d& T_cw = m_camera_poses[camera_index];
    Eigen::Matrix3d R_cw = T_cw.block<3,3>(0,0);
    Eigen::Vector3d t_cw = T_cw.block<3,1>(0,3);
    
    // Compute world-to-camera (inverse)
    Eigen::Matrix3d R_wc = R_cw.transpose();
    Eigen::Vector3d t_wc = -R_wc * t_cw;
    
    int num_rendered = 0;
    
    // Project all Gaussians
    for (const auto& g : m_gaussians) {
        // Transform position to camera frame
        Eigen::Vector3d p_world = g.position.cast<double>();
        Eigen::Vector3d p_cam = R_wc * p_world + t_wc;
        
        // Check if in front of camera
        if (p_cam.z() <= 0.0) continue;
        
        // Project to image plane
        float x = m_cam_fx * (p_cam.x() / p_cam.z()) + (width / 2.0f);
        float y = m_cam_fy * (p_cam.y() / p_cam.z()) + (height / 2.0f);
        
        // Check if in image bounds
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        
        // Convert SH DC to RGB: RGB = max(0, SH_C0 * f_dc + 0.5)
        const float SH_C0 = 0.28209479177387814f;
        float r = std::max(0.0f, g.sh_dc.x() * SH_C0 + 0.5f);
        float g_val = std::max(0.0f, g.sh_dc.y() * SH_C0 + 0.5f);
        float b = std::max(0.0f, g.sh_dc.z() * SH_C0 + 0.5f);
        float opacity = 1.0f / (1.0f + std::exp(-g.opacity));
        
        // Filter out very transparent points to match rasterizer behavior
        if (opacity < 0.05f) continue;
        
        // Convert to 0-255 range
        cv::Vec3b color(
            static_cast<uint8_t>(b * 255),
            static_cast<uint8_t>(g_val * 255),
            static_cast<uint8_t>(r * 255)
        );
        
        // Simple point rendering - just draw a single pixel
        int px = static_cast<int>(x);
        int py = static_cast<int>(y);
        
        num_rendered++;
        
        // Draw single pixel point
        if (px >= 0 && px < width && py >= 0 && py < height) {
            cv::Vec3b& pixel = image.at<cv::Vec3b>(py, px);
            pixel[0] = static_cast<uint8_t>(pixel[0] * (1 - opacity) + color[0] * opacity);
            pixel[1] = static_cast<uint8_t>(pixel[1] * (1 - opacity) + color[1] * opacity);
            pixel[2] = static_cast<uint8_t>(pixel[2] * (1 - opacity) + color[2] * opacity);
        }
    }
    
    std::cout << "[GaussianViewer] Point rendering: " << num_rendered << " points drawn" << std::endl;
    
    return image;
}


cv::Mat GaussianViewer::RenderCPUGaussianSplatting(int camera_index, int width, int height, float scale_modifier) {
    if (camera_index < 0 || camera_index >= m_camera_poses.size()) {
        std::cerr << "[GaussianViewer] Invalid camera index: " << camera_index << std::endl;
        return cv::Mat();
    }
    
    if (!m_cpu_renderer) {
        std::cerr << "[GaussianViewer] CPU Renderer not initialized!" << std::endl;
        return cv::Mat();
    }
    
    std::cout << "[GaussianViewer] === Rendering CPU Gaussian Splatting view " << camera_index << " ===" << std::endl;
    
    // Get camera pose (camera-to-world)
    const Eigen::Matrix4d& T_cw = m_camera_poses[camera_index];
    
    // Create camera object
    gaussian_splat_engine::CameraIntrinsics intrinsics;
    intrinsics.width = width;
    intrinsics.height = height;
    intrinsics.fx = m_cam_fx;
    intrinsics.fy = m_cam_fy;
    intrinsics.cx = width / 2.0f;
    intrinsics.cy = height / 2.0f;
    
    gaussian_splat_engine::Camera camera(intrinsics);
    camera.SetPose(T_cw);
    
    // Render using CPU
    std::cout << "[GaussianViewer] Calling CPU Render()..." << std::endl;
    cv::Mat image = m_cpu_renderer->Render(m_model, camera, width, height, scale_modifier);
    std::cout << "[GaussianViewer] CPU Render complete." << std::endl;
    
    return image;
}
