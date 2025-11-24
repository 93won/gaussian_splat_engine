/**
 * @file      JSONUtils.h
 * @brief     JSON parsing utilities for camera data
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 */

#pragma once

#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace json_utils {

/**
 * @brief Camera info from cameras.json (3DGS format)
 */
struct CameraInfo {
    int id;
    std::string img_name;
    int width;
    int height;
    Eigen::Vector3f position;      // Camera center in world
    Eigen::Matrix3f rotation;      // Camera-to-world rotation matrix
    float fx;
    float fy;
    
    CameraInfo() 
        : id(-1)
        , width(0)
        , height(0)
        , position(Eigen::Vector3f::Zero())
        , rotation(Eigen::Matrix3f::Identity())
        , fx(0.0f)
        , fy(0.0f) {}
};

/**
 * @brief Load cameras from cameras.json file (3DGS format)
 * @param json_path Path to cameras.json
 * @param cameras Output vector of camera info
 * @return true if successful, false otherwise
 */
bool LoadCamerasJSON(const std::string& json_path, std::vector<CameraInfo>& cameras);

} // namespace json_utils
