/**
 * @file      Camera.h
 * @brief     Camera class for pinhole camera model with intrinsics and extrinsics
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 */

#ifndef GAUSSIAN_SPLAT_ENGINE_CAMERA_H
#define GAUSSIAN_SPLAT_ENGINE_CAMERA_H

#include <Eigen/Dense>
#include <string>

namespace gaussian_splat_engine {

/**
 * @brief Camera intrinsic parameters structure
 */
struct CameraIntrinsics {
    double fx, fy;      // Focal lengths
    double cx, cy;      // Principal point
    int width, height;  // Image dimensions
    
    CameraIntrinsics() 
        : fx(500.0), fy(500.0), cx(320.0), cy(240.0), width(640), height(480) {}
};

/**
 * @brief Camera class for pinhole camera model
 * 
 * Handles camera intrinsic and extrinsic parameters, and provides
 * projection/unprojection functions for 3D-2D transformations.
 */
class Camera {
public:
    /**
     * @brief Default constructor with identity pose
     */
    Camera();
    
    /**
     * @brief Constructor with intrinsics from config file
     * @param config_file Path to YAML configuration file
     */
    explicit Camera(const std::string& config_file);
    
    /**
     * @brief Constructor with intrinsics structure
     * @param intrinsics Camera intrinsic parameters
     */
    explicit Camera(const CameraIntrinsics& intrinsics);
    
    /**
     * @brief Destructor
     */
    ~Camera() = default;
    
    // ==================== Intrinsic Parameters ====================
    
    /**
     * @brief Set camera intrinsic parameters
     */
    void SetIntrinsics(double fx, double fy, double cx, double cy, 
                      int width, int height);
    
    /**
     * @brief Get focal length in x direction
     */
    double fx() const { return m_fx; }
    
    /**
     * @brief Get focal length in y direction
     */
    double fy() const { return m_fy; }
    
    /**
     * @brief Get principal point x coordinate
     */
    double cx() const { return m_cx; }
    
    /**
     * @brief Get principal point y coordinate
     */
    double cy() const { return m_cy; }
    
    /**
     * @brief Get image width
     */
    int width() const { return m_width; }
    
    /**
     * @brief Get image height
     */
    int height() const { return m_height; }
    
    /**
     * @brief Get camera intrinsic matrix K (3x3)
     */
    Eigen::Matrix3d GetIntrinsicMatrix() const;
    
    // ==================== Extrinsic Parameters ====================
    
    /**
     * @brief Set camera pose (world to camera transform)
     * @param R Rotation matrix (3x3)
     * @param t Translation vector (3x1)
     */
    void SetPose(const Eigen::Matrix3d& R, const Eigen::Vector3d& t);
    
    /**
     * @brief Set camera pose from quaternion and translation
     * @param q Rotation quaternion
     * @param t Translation vector
     */
    void SetPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t);
    
    /**
     * @brief Set camera pose from 4x4 transformation matrix
     * @param T Transformation matrix (world to camera)
     */
    void SetPose(const Eigen::Matrix4d& T);
    
    /**
     * @brief Get 4x4 pose matrix (camera to world)
     * @return Camera pose transformation matrix
     */
    Eigen::Matrix4d GetPose() const;
    
    /**
     * @brief Get rotation matrix (world to camera)
     */
    Eigen::Matrix3d GetRotation() const { return m_R; }
    
    /**
     * @brief Get translation vector (world to camera)
     */
    Eigen::Vector3d GetTranslation() const { return m_t; }
    
    /**
     * @brief Get full transformation matrix (4x4)
     */
    Eigen::Matrix4d GetTransformMatrix() const;
    
    /**
     * @brief Get camera center in world coordinates
     */
    Eigen::Vector3d GetCameraCenter() const;
    
    // ==================== Projection Functions ====================
    
    /**
     * @brief Project 3D point in world coordinates to 2D image pixel
     * @param point_world 3D point in world frame
     * @return 2D pixel coordinates (u, v)
     */
    Eigen::Vector2d Project(const Eigen::Vector3d& point_world) const;
    
    /**
     * @brief Project 3D point in camera coordinates to 2D image pixel
     * @param point_camera 3D point in camera frame
     * @return 2D pixel coordinates (u, v)
     */
    Eigen::Vector2d ProjectCameraFrame(const Eigen::Vector3d& point_camera) const;
    
    /**
     * @brief Unproject 2D pixel to 3D ray in world coordinates
     * @param pixel 2D pixel coordinates (u, v)
     * @param depth Depth value (optional, default=1.0 for direction only)
     * @return 3D point in world frame
     */
    Eigen::Vector3d Unproject(const Eigen::Vector2d& pixel, double depth = 1.0) const;
    
    /**
     * @brief Unproject 2D pixel to 3D point in camera coordinates
     * @param pixel 2D pixel coordinates (u, v)
     * @param depth Depth value
     * @return 3D point in camera frame
     */
    Eigen::Vector3d UnprojectCameraFrame(const Eigen::Vector2d& pixel, double depth) const;
    
    /**
     * @brief Transform 3D point from world to camera coordinates
     * @param point_world 3D point in world frame
     * @return 3D point in camera frame
     */
    Eigen::Vector3d WorldToCamera(const Eigen::Vector3d& point_world) const;
    
    /**
     * @brief Transform 3D point from camera to world coordinates
     * @param point_camera 3D point in camera frame
     * @return 3D point in world frame
     */
    Eigen::Vector3d CameraToWorld(const Eigen::Vector3d& point_camera) const;
    
    /**
     * @brief Check if pixel is within image bounds
     * @param pixel 2D pixel coordinates
     * @param border Border margin (default=0)
     * @return true if pixel is valid
     */
    bool IsPixelValid(const Eigen::Vector2d& pixel, int border = 0) const;
    
    // ==================== I/O Functions ====================
    
    /**
     * @brief Load camera parameters from YAML config file
     * @param config_file Path to configuration file
     * @return true if loading succeeded
     */
    bool LoadFromConfig(const std::string& config_file);
    
    /**
     * @brief Print camera parameters
     */
    void PrintInfo() const;

private:
    // Intrinsic parameters
    double m_fx, m_fy;      // Focal length
    double m_cx, m_cy;      // Principal point
    int m_width, m_height;  // Image dimensions
    
    // Extrinsic parameters (world to camera transform: p_camera = R * p_world + t)
    Eigen::Matrix3d m_R;    // Rotation matrix
    Eigen::Vector3d m_t;    // Translation vector
    
    // Cached inverse (camera to world)
    Eigen::Matrix3d m_R_inv;
    Eigen::Vector3d m_t_inv;
    
    /**
     * @brief Update cached inverse transformation
     */
    void UpdateInverse();
};

} // namespace gaussian_splat_engine

#endif // GAUSSIAN_SPLAT_ENGINE_CAMERA_H
