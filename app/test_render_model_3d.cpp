/**
 * @file      test_render_model_3d.cpp
 * @brief     Test GaussianRasterizer with 3D visualization
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 */

#include "GaussianModel.h"
#include "Camera.h"
#include "GaussianRasterizer.h"
#include "../src/util/PLYUtils.h"
#include "../src/util/JSONUtils.h"
#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace gaussian_splat_engine;
using namespace gaussian_rasterizer;

// Helper function to draw coordinate axes
void DrawCoordinateAxes(float length = 0.5f) {
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    
    // X axis - Red
    glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(length, 0.0f, 0.0f);
    
    // Y axis - Green
    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, length, 0.0f);
    
    // Z axis - Blue
    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, length);
    
    glEnd();
}

// Draw camera frustum
void DrawCameraFrustum(const json_utils::CameraInfo& cam_info, float frustum_depth = 0.2f) {
    Eigen::Matrix3f R = cam_info.rotation;
    Eigen::Vector3f t = cam_info.position;
    
    // Camera parameters
    float fx = cam_info.fx;
    float fy = cam_info.fy;
    int width = cam_info.width;
    int height = cam_info.height;
    
    // Frustum corners in camera frame
    float half_width = (width / 2.0f) / fx * frustum_depth;
    float half_height = (height / 2.0f) / fy * frustum_depth;
    
    Eigen::Vector3f corners_cam[4];
    corners_cam[0] = Eigen::Vector3f(-half_width, -half_height, frustum_depth);
    corners_cam[1] = Eigen::Vector3f(half_width, -half_height, frustum_depth);
    corners_cam[2] = Eigen::Vector3f(half_width, half_height, frustum_depth);
    corners_cam[3] = Eigen::Vector3f(-half_width, half_height, frustum_depth);
    
    // Transform to world frame
    Eigen::Vector3f corners_world[4];
    for (int i = 0; i < 4; i++) {
        corners_world[i] = R * corners_cam[i] + t;
    }
    
    // Draw frustum edges
    glLineWidth(1.5f);
    glColor3f(0.0f, 1.0f, 0.0f); // Green
    glBegin(GL_LINES);
    for (int i = 0; i < 4; i++) {
        glVertex3f(t.x(), t.y(), t.z());
        glVertex3f(corners_world[i].x(), corners_world[i].y(), corners_world[i].z());
    }
    glEnd();
    
    // Draw image plane
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 4; i++) {
        glVertex3f(corners_world[i].x(), corners_world[i].y(), corners_world[i].z());
    }
    glEnd();
    
    // Draw camera center
    glColor3f(1.0f, 1.0f, 0.0f); // Yellow
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    glVertex3f(t.x(), t.y(), t.z());
    glEnd();
}

// Draw all camera trajectory
void DrawCameraTrajectory(const std::vector<json_utils::CameraInfo>& cameras) {
    if (cameras.empty()) return;
    
    // Draw trajectory line
    glLineWidth(2.0f);
    glColor3f(1.0f, 0.0f, 0.0f); // Red
    glBegin(GL_LINE_STRIP);
    for (const auto& cam : cameras) {
        Eigen::Vector3f pos = cam.position;
        glVertex3f(pos.x(), pos.y(), pos.z());
    }
    glEnd();
}

// Draw Gaussian points
void DrawGaussianPoints(const std::vector<Gaussian>& gaussians, float point_size = 2.0f) {
    if (gaussians.empty()) return;
    
    glPointSize(point_size);
    glBegin(GL_POINTS);
    
    const float SH_C0 = 0.28209479177387814f;
    
    for (const auto& g : gaussians) {
        // Sigmoid for opacity
        float opacity = 1.0f / (1.0f + std::exp(-g.opacity));
        if (opacity < 0.05f) continue;
        
        // SH DC to RGB
        float r = std::max(0.0f, std::min(1.0f, g.sh_dc.x() * SH_C0 + 0.5f));
        float g_val = std::max(0.0f, std::min(1.0f, g.sh_dc.y() * SH_C0 + 0.5f));
        float b = std::max(0.0f, std::min(1.0f, g.sh_dc.z() * SH_C0 + 0.5f));
        
        glColor4f(r, g_val, b, opacity);
        glVertex3f(g.position.x(), g.position.y(), g.position.z());
    }
    
    glEnd();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <ply_file> <cameras_json>" << std::endl;
        std::cerr << "Example: " << argv[0] << " point_cloud.ply cameras.json" << std::endl;
        return 1;
    }
    
    std::string ply_path = argv[1];
    std::string cameras_path = argv[2];
    
    // Load Gaussians
    std::cout << "Loading Gaussians from: " << ply_path << std::endl;
    std::vector<Gaussian> gaussians;
    GaussianModel model;
    if (!model.LoadFromPLY(ply_path)) {
        std::cerr << "Failed to load PLY file!" << std::endl;
        return 1;
    }
    gaussians = model.GetGaussians();
    std::cout << "Loaded " << gaussians.size() << " Gaussians" << std::endl;
    
    // Load cameras
    std::cout << "Loading cameras from: " << cameras_path << std::endl;
    std::vector<json_utils::CameraInfo> cameras;
    if (!json_utils::LoadCamerasJSON(cameras_path, cameras)) {
        std::cerr << "Failed to load cameras!" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << cameras.size() << " cameras" << std::endl;
    
    if (cameras.empty()) {
        std::cerr << "No cameras found!" << std::endl;
        return 1;
    }
    
    // Render resolution
    const int render_width = 1280;
    const int render_height = 853;
    
    std::cout << "Render resolution: " << render_width << "x" << render_height << std::endl;
    
    // Create OpenCV window for rendered image
    cv::namedWindow("Rendered Image", cv::WINDOW_NORMAL);
    cv::resizeWindow("Rendered Image", render_width, render_height);
    
    // Create Pangolin window for 3D view
    const int win_width = 1600;
    const int win_height = 900;
    pangolin::CreateWindowAndBind("3D Visualization", win_width, win_height);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    // Define camera for 3D view
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(win_width, win_height, 500, 500, win_width/2, win_height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -2, -5, 0, 0, 0, pangolin::AxisY)
    );
    
    // Layout: UI panel on left (250px), rest is 3D view
    const int ui_panel_width = 250;
    
    // UI panel at left
    pangolin::CreatePanel("ui")
        .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(ui_panel_width));
    
    // 3D view on the right side
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(ui_panel_width), 1.0)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
    // UI variables
    pangolin::Var<int> ui_camera_id("ui.Camera ID", 0, 0, cameras.size() - 1);
    pangolin::Var<bool> btn_next_cam("ui.Next Camera", false, false);
    pangolin::Var<bool> btn_prev_cam("ui.Prev Camera", false, false);
    pangolin::Var<float> ui_scale_modifier("ui.Scale Modifier", 1.0f, 0.1f, 5.0f);
    pangolin::Var<int> ui_sh_degree("ui.SH Degree", 3, 0, 3);
    pangolin::Var<float> ui_alpha_threshold("ui.Alpha Threshold", 1.0f/255.0f, 0.0f, 0.5f);
    pangolin::Var<bool> ui_show_points("ui.Show 3D Points", true, true);
    pangolin::Var<bool> ui_show_trajectory("ui.Show Trajectory", true, true);
    pangolin::Var<bool> ui_show_axes("ui.Show Axes", true, true);
    pangolin::Var<bool> ui_show_current_cam("ui.Show Active Camera", true, true);
    pangolin::Var<float> ui_point_size("ui.Point Size", 2.0f, 1.0f, 10.0f);
    pangolin::Var<bool> ui_debug("ui.Debug", false, true);
    
    // Create rasterizer
    GaussianRasterizer rasterizer;
    
    int prev_cam_id = -1;
    float prev_scale = -1.0f;
    int prev_sh_deg = -1;
    float prev_alpha_threshold = -1.0f;
    bool need_render = true;
    
    // OpenCV image for display
    cv::Mat rendered_image;
    
    std::cout << "\n========== Controls ==========\n";
    std::cout << "  Separate Windows:\n";
    std::cout << "    - '3D Visualization': 3D points, cameras, trajectory\n";
    std::cout << "    - 'Rendered Image': Rasterized output\n";
    std::cout << "  Navigation:\n";
    std::cout << "    - Next/Prev Camera buttons: Switch viewpoint\n";
    std::cout << "    - Camera ID slider: Jump to specific camera\n";
    std::cout << "  3D View Controls:\n";
    std::cout << "    - Drag: Rotate view\n";
    std::cout << "    - Right-click drag: Pan\n";
    std::cout << "    - Scroll: Zoom\n";
    std::cout << "  UI Panel: Adjust camera/scale/SH\n";
    std::cout << "==============================\n" << std::endl;
    
    // Main loop
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Handle camera navigation buttons
        if (pangolin::Pushed(btn_next_cam)) {
            ui_camera_id = (ui_camera_id + 1) % cameras.size();
            std::cout << "[Navigation] Next camera -> " << (int)ui_camera_id << std::endl;
        }
        if (pangolin::Pushed(btn_prev_cam)) {
            ui_camera_id = (ui_camera_id - 1 + cameras.size()) % cameras.size();
            std::cout << "[Navigation] Previous camera -> " << (int)ui_camera_id << std::endl;
        }
        
        int cam_id = std::max(0, std::min((int)cameras.size() - 1, (int)ui_camera_id));
        
        // Check if parameters changed
        if (cam_id != prev_cam_id || ui_scale_modifier != prev_scale || 
            ui_sh_degree != prev_sh_deg || ui_alpha_threshold != prev_alpha_threshold) {
            need_render = true;
            prev_cam_id = cam_id;
            prev_scale = ui_scale_modifier;
            prev_sh_deg = ui_sh_degree;
            prev_alpha_threshold = ui_alpha_threshold;
        }
        
        // Render if needed
        if (need_render) {
            const auto& cam_info = cameras[cam_id];
            
            std::cout << "\n[Rendering] Camera " << cam_id << ", Scale " << ui_scale_modifier 
                      << ", SH degree " << ui_sh_degree << std::endl;
            
            // Setup raster settings
            RasterSettings settings;
            settings.image_width = render_width;
            settings.image_height = render_height;
            
            // Camera intrinsics
            float scale_x = (float)render_width / cam_info.width;
            float scale_y = (float)render_height / cam_info.height;
            settings.focal_x = cam_info.fx * scale_x;
            settings.focal_y = cam_info.fy * scale_y;
            
            // FOV
            settings.tan_fovx = (render_width / 2.0f) / settings.focal_x;
            settings.tan_fovy = (render_height / 2.0f) / settings.focal_y;
            
            // Camera pose (world-to-camera)
            // cam_info has camera-to-world rotation and camera position in world
            Eigen::Matrix3f R_cw = cam_info.rotation;      // Camera-to-world
            Eigen::Vector3f t_w = cam_info.position;       // Camera position in world
            Eigen::Matrix3f R_wc = R_cw.transpose();       // World-to-camera
            Eigen::Vector3f t_wc = -R_wc * t_w;            // Translation in world-to-camera
            
            std::cout << "[Debug] Camera " << cam_id << " position: (" 
                      << t_w.x() << ", " << t_w.y() << ", " << t_w.z() << ")" << std::endl;
            std::cout << "[Debug] Camera rotation (first row): (" 
                      << R_cw(0,0) << ", " << R_cw(0,1) << ", " << R_cw(0,2) << ")" << std::endl;
            
            // CRITICAL: Set camera position for SH evaluation (viewing direction)
            settings.campos = t_w;  // Camera position in world coordinates
            
            settings.viewmatrix = Eigen::Matrix4f::Identity();
            settings.viewmatrix.block<3, 3>(0, 0) = R_wc;
            settings.viewmatrix.block<3, 1>(0, 3) = t_wc;
            
            // Projection matrix (OpenGL style)
            float znear = 0.01f;
            float zfar = 100.0f;
            Eigen::Matrix4f projection = Eigen::Matrix4f::Zero();
            projection(0, 0) = 2.0f * settings.focal_x / render_width;
            projection(1, 1) = 2.0f * settings.focal_y / render_height;
            projection(2, 2) = -(zfar + znear) / (zfar - znear);
            projection(2, 3) = -2.0f * zfar * znear / (zfar - znear);
            projection(3, 2) = -1.0f;
            
            // CRITICAL: projmatrix must be proj * view (transforms world -> clip space)
            settings.projmatrix = projection * settings.viewmatrix;
            
            settings.scale_modifier = ui_scale_modifier;
            settings.sh_degree = ui_sh_degree;
            settings.alpha_threshold = ui_alpha_threshold;
            
            Eigen::Vector3f bg(0.0f, 0.0f, 0.0f);
            settings.bg_color[0] = bg.x();
            settings.bg_color[1] = bg.y();
            settings.bg_color[2] = bg.z();
            
            // Render
            std::cout << "[Rendering] Starting render..." << std::endl;
            RenderOutput output = rasterizer.Render(gaussians, settings);
            std::cout << "[Rendering] Render complete!" << std::endl;
            
            // Convert to OpenCV Mat (BGR format)
            std::cout << "[Rendering] Converting to OpenCV image..." << std::endl;
            rendered_image = cv::Mat(render_height, render_width, CV_8UC3);
            for (int y = 0; y < render_height; ++y) {
                for (int x = 0; x < render_width; ++x) {
                    int idx = y * render_width + x;
                    // Convert RGB to BGR for OpenCV
                    rendered_image.at<cv::Vec3b>(y, x)[0] = static_cast<unsigned char>(output.color[idx * 3 + 2] * 255.0f); // B
                    rendered_image.at<cv::Vec3b>(y, x)[1] = static_cast<unsigned char>(output.color[idx * 3 + 1] * 255.0f); // G
                    rendered_image.at<cv::Vec3b>(y, x)[2] = static_cast<unsigned char>(output.color[idx * 3 + 0] * 255.0f); // R
                }
            }
            
            // Flip vertically because OpenGL y-axis is bottom-to-top, but image is top-to-bottom
            cv::flip(rendered_image, rendered_image, 0);
            
            // Display with OpenCV
            cv::imshow("Rendered Image", rendered_image);
            std::cout << "[Rendering] Done!" << std::endl;
            
            need_render = false;
        }
        
        // Process OpenCV events (must be called to display window)
        cv::waitKey(1);
        
        // Draw 3D view
        d_cam.Activate(s_cam);
        
        if (ui_show_axes) {
            DrawCoordinateAxes(0.5f);
        }
        
        if (ui_show_points) {
            DrawGaussianPoints(gaussians, ui_point_size);
        }
        
        if (ui_show_trajectory) {
            DrawCameraTrajectory(cameras);
        }
        
        // Draw current active camera frustum only
        if (ui_show_current_cam) {
            glLineWidth(3.0f);
            DrawCameraFrustum(cameras[cam_id], 0.25f);
        }
        
        pangolin::FinishFrame();
    }
    
    std::cout << "Exiting..." << std::endl;
    cv::destroyAllWindows();
    
    return 0;
}
