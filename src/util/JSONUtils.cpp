/**
 * @file      JSONUtils.cpp
 * @brief     JSON parsing utilities for camera data
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-24
 */

#include "JSONUtils.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

namespace json_utils {

// Simple JSON parser for cameras.json (no external library needed)
bool LoadCamerasJSON(const std::string& json_path, std::vector<CameraInfo>& cameras) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        std::cerr << "[JSONUtils] Cannot open JSON file: " << json_path << std::endl;
        return false;
    }
    
    cameras.clear();
    
    // Read entire file
    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();
    
    // Find array start
    size_t array_start = content.find('[');
    if (array_start == std::string::npos) {
        std::cerr << "[JSONUtils] Invalid JSON format: no array found" << std::endl;
        return false;
    }
    
    // Parse each camera object (simple parsing)
    size_t pos = array_start + 1;
    while (pos < content.size()) {
        // Find next object
        size_t obj_start = content.find('{', pos);
        if (obj_start == std::string::npos) break;
        
        size_t obj_end = content.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        
        std::string obj = content.substr(obj_start, obj_end - obj_start + 1);
        
        CameraInfo cam;
        
        // Parse id
        size_t id_pos = obj.find("\"id\":");
        if (id_pos != std::string::npos) {
            sscanf(obj.c_str() + id_pos + 5, "%d", &cam.id);
        }
        
        // Parse img_name
        size_t name_pos = obj.find("\"img_name\":");
        if (name_pos != std::string::npos) {
            size_t name_start = obj.find('\"', name_pos + 11) + 1;
            size_t name_end = obj.find('\"', name_start);
            cam.img_name = obj.substr(name_start, name_end - name_start);
        }
        
        // Parse width, height
        size_t width_pos = obj.find("\"width\":");
        if (width_pos != std::string::npos) {
            sscanf(obj.c_str() + width_pos + 8, "%d", &cam.width);
        }
        
        size_t height_pos = obj.find("\"height\":");
        if (height_pos != std::string::npos) {
            sscanf(obj.c_str() + height_pos + 9, "%d", &cam.height);
        }
        
        // Parse position [x, y, z]
        size_t pos_pos = obj.find("\"position\":");
        if (pos_pos != std::string::npos) {
            size_t pos_arr_start = obj.find('[', pos_pos);
            float x, y, z;
            sscanf(obj.c_str() + pos_arr_start + 1, "%f, %f, %f", &x, &y, &z);
            cam.position = Eigen::Vector3f(x, y, z);
        }
        
        // Parse rotation [[...], [...], [...]]
        size_t rot_pos = obj.find("\"rotation\":");
        if (rot_pos != std::string::npos) {
            size_t rot_arr_start = obj.find('[', rot_pos);
            size_t row0_start = obj.find('[', rot_arr_start + 1);
            size_t row1_start = obj.find('[', row0_start + 1);
            size_t row2_start = obj.find('[', row1_start + 1);
            
            float r00, r01, r02, r10, r11, r12, r20, r21, r22;
            sscanf(obj.c_str() + row0_start + 1, "%f, %f, %f", &r00, &r01, &r02);
            sscanf(obj.c_str() + row1_start + 1, "%f, %f, %f", &r10, &r11, &r12);
            sscanf(obj.c_str() + row2_start + 1, "%f, %f, %f", &r20, &r21, &r22);
            
            cam.rotation << r00, r01, r02,
                            r10, r11, r12,
                            r20, r21, r22;
        }
        
        // Parse fx, fy
        size_t fx_pos = obj.find("\"fx\":");
        if (fx_pos != std::string::npos) {
            sscanf(obj.c_str() + fx_pos + 5, "%f", &cam.fx);
        }
        
        size_t fy_pos = obj.find("\"fy\":");
        if (fy_pos != std::string::npos) {
            sscanf(obj.c_str() + fy_pos + 5, "%f", &cam.fy);
        }
        
        cameras.push_back(cam);
        pos = obj_end + 1;
    }
    
    std::cout << "[JSONUtils] Loaded " << cameras.size() << " cameras from " << json_path << std::endl;
    return !cameras.empty();
}

} // namespace json_utils
