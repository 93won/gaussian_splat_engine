/**
 * @file      Camera.cpp
 * @brief     Implementation of Camera class
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 */

#include "database/Camera.h"
#include <iostream>
#include <iomanip>

namespace gaussian_splat_engine {

Camera::Camera() 
    : m_fx(500.0), m_fy(500.0), m_cx(320.0), m_cy(240.0),
      m_width(640), m_height(480),
      m_R(Eigen::Matrix3d::Identity()),
      m_t(Eigen::Vector3d::Zero()) {
    UpdateInverse();
}

Camera::Camera(const std::string& config_file) : Camera() {
    LoadFromConfig(config_file);
}

Camera::Camera(const CameraIntrinsics& intrinsics) : Camera() {
    SetIntrinsics(intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                  intrinsics.width, intrinsics.height);
}

// ==================== Intrinsic Parameters ====================

void Camera::SetIntrinsics(double fx, double fy, double cx, double cy, 
                          int width, int height) {
    m_fx = fx;
    m_fy = fy;
    m_cx = cx;
    m_cy = cy;
    m_width = width;
    m_height = height;
}

Eigen::Matrix3d Camera::GetIntrinsicMatrix() const {
    Eigen::Matrix3d K;
    K << m_fx,  0.0,  m_cx,
         0.0,  m_fy,  m_cy,
         0.0,  0.0,   1.0;
    return K;
}

// ==================== Extrinsic Parameters ====================

void Camera::SetPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
    m_R = R;
    m_t = t;
    UpdateInverse();
}

void Camera::SetPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    m_R = q.toRotationMatrix();
    m_t = t;
    UpdateInverse();
}

void Camera::SetPose(const Eigen::Matrix4d& T) {
    m_R = T.block<3, 3>(0, 0);
    m_t = T.block<3, 1>(0, 3);
    UpdateInverse();
}

Eigen::Matrix4d Camera::GetPose() const {
    // Return camera-to-world transform (inverse of world-to-camera)
    Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
    T_cw.block<3, 3>(0, 0) = m_R_inv;  // R^T
    T_cw.block<3, 1>(0, 3) = m_t_inv;  // -R^T * t
    return T_cw;
}

Eigen::Matrix4d Camera::GetTransformMatrix() const {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = m_R;
    T.block<3, 1>(0, 3) = m_t;
    return T;
}

Eigen::Vector3d Camera::GetCameraCenter() const {
    // Camera center in world frame: C = -R^T * t
    return m_t_inv;
}

void Camera::UpdateInverse() {
    // Camera to world transform: p_world = R^T * (p_camera - t) = R^T * p_camera - R^T * t
    m_R_inv = m_R.transpose();
    m_t_inv = -m_R_inv * m_t;
}

// ==================== Projection Functions ====================

Eigen::Vector2d Camera::Project(const Eigen::Vector3d& point_world) const {
    // Transform to camera frame
    Eigen::Vector3d point_camera = WorldToCamera(point_world);
    
    // Project to image plane
    return ProjectCameraFrame(point_camera);
}

Eigen::Vector2d Camera::ProjectCameraFrame(const Eigen::Vector3d& point_camera) const {
    // Pinhole projection: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
    double z_inv = 1.0 / point_camera.z();
    double u = m_fx * point_camera.x() * z_inv + m_cx;
    double v = m_fy * point_camera.y() * z_inv + m_cy;
    
    return Eigen::Vector2d(u, v);
}

Eigen::Vector3d Camera::Unproject(const Eigen::Vector2d& pixel, double depth) const {
    // Unproject to camera frame
    Eigen::Vector3d point_camera = UnprojectCameraFrame(pixel, depth);
    
    // Transform to world frame
    return CameraToWorld(point_camera);
}

Eigen::Vector3d Camera::UnprojectCameraFrame(const Eigen::Vector2d& pixel, double depth) const {
    // Inverse pinhole projection: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
    double x = (pixel.x() - m_cx) * depth / m_fx;
    double y = (pixel.y() - m_cy) * depth / m_fy;
    double z = depth;
    
    return Eigen::Vector3d(x, y, z);
}

Eigen::Vector3d Camera::WorldToCamera(const Eigen::Vector3d& point_world) const {
    // p_camera = R * p_world + t
    return m_R * point_world + m_t;
}

Eigen::Vector3d Camera::CameraToWorld(const Eigen::Vector3d& point_camera) const {
    // p_world = R^T * (p_camera - t)
    return m_R_inv * (point_camera - m_t);
}

bool Camera::IsPixelValid(const Eigen::Vector2d& pixel, int border) const {
    return pixel.x() >= border && pixel.x() < m_width - border &&
           pixel.y() >= border && pixel.y() < m_height - border;
}

// ==================== I/O Functions ====================

bool Camera::LoadFromConfig(const std::string& config_file) {
    std::cerr << "[Camera] LoadFromConfig is deprecated. ConfigUtils has been removed." << std::endl;
    std::cerr << "[Camera] Please use Camera(CameraIntrinsics) constructor instead." << std::endl;
    (void)config_file;  // Suppress unused parameter warning
    return false;
}

void Camera::PrintInfo() const {
    std::cout << "\n========== Camera Parameters ==========" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    
    // Intrinsics
    std::cout << "Intrinsics:" << std::endl;
    std::cout << "  - Image size: " << m_width << "x" << m_height << std::endl;
    std::cout << "  - Focal length: fx=" << m_fx << ", fy=" << m_fy << std::endl;
    std::cout << "  - Principal point: cx=" << m_cx << ", cy=" << m_cy << std::endl;
    
    // Extrinsics
    std::cout << "\nExtrinsics (World to Camera):" << std::endl;
    std::cout << "  - Rotation:\n" << m_R << std::endl;
    std::cout << "  - Translation: [" << m_t.transpose() << "]" << std::endl;
    
    // Camera center
    Eigen::Vector3d center = GetCameraCenter();
    std::cout << "  - Camera center (world): [" << center.transpose() << "]" << std::endl;
    
    std::cout << "=======================================" << std::endl;
}

} // namespace gaussian_splat_engine
