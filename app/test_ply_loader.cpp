/**
 * @file      test_ply_loader.cpp
 * @brief     Test application for PLY loader with visualization
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 */

#include "GaussianModel.h"
#include "GaussianViewer.h"
#include <iostream>
#include <iomanip>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <ply_file_path>" << std::endl;
        std::cerr << "Example: " << argv[0] << " ../results/datasets_TUM-RGBD/2025-11-23-11-11-27/point_cloud/final/point_cloud.ply" << std::endl;
        return 1;
    }
    
    std::string ply_path = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Gaussian Splatting PLY Loader Test   " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Load PLY file
    std::cout << "Loading PLY file: " << ply_path << std::endl;
    
    GaussianModel model;
    if (!model.LoadFromPLY(ply_path)) {
        std::cerr << "Failed to load PLY file!" << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Statistics                            " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total Gaussians: " << model.GetNumGaussians() << std::endl;
    std::cout << std::endl;
    
    // Print first 10 Gaussians
    const auto& gaussians = model.GetGaussians();
    size_t num_to_print = std::min(size_t(10), gaussians.size());
    
    std::cout << "========================================" << std::endl;
    std::cout << "  First " << num_to_print << " Gaussians" << std::endl;
    std::cout << "========================================" << std::endl;
    
    for (size_t i = 0; i < num_to_print; ++i) {
        const auto& g = gaussians[i];
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "[" << i << "]" << std::endl;
        std::cout << "  Position: (" << g.position.x() << ", " << g.position.y() << ", " << g.position.z() << ")" << std::endl;
        std::cout << "  Scale:    (" << g.scale.x() << ", " << g.scale.y() << ", " << g.scale.z() << ")" << std::endl;
        std::cout << "  Rotation: (" << g.rotation.w() << ", " << g.rotation.x() << ", " << g.rotation.y() << ", " << g.rotation.z() << ")" << std::endl;
        std::cout << "  Color DC: (" << g.color_dc.x() << ", " << g.color_dc.y() << ", " << g.color_dc.z() << ")" << std::endl;
        std::cout << "  Opacity:  " << g.opacity << std::endl;
        std::cout << std::endl;
    }
    
    // Compute statistics
    Eigen::Vector3f min_pos = gaussians[0].position;
    Eigen::Vector3f max_pos = gaussians[0].position;
    float min_opacity = gaussians[0].opacity;
    float max_opacity = gaussians[0].opacity;
    
    for (const auto& g : gaussians) {
        min_pos = min_pos.cwiseMin(g.position);
        max_pos = max_pos.cwiseMax(g.position);
        min_opacity = std::min(min_opacity, g.opacity);
        max_opacity = std::max(max_opacity, g.opacity);
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "  Bounding Box & Range                  " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Position Min: (" << min_pos.x() << ", " << min_pos.y() << ", " << min_pos.z() << ")" << std::endl;
    std::cout << "Position Max: (" << max_pos.x() << ", " << max_pos.y() << ", " << max_pos.z() << ")" << std::endl;
    std::cout << "Opacity Range: [" << min_opacity << ", " << max_opacity << "]" << std::endl;
    std::cout << std::endl;
    
    std::cout << "PLY loader test completed successfully!" << std::endl;
    std::cout << std::endl;
    
    // Visualize with Pangolin
    std::cout << "========================================" << std::endl;
    std::cout << "  Visualization                         " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Initializing viewer..." << std::endl;
    
    GaussianViewer viewer;
    if (!viewer.Initialize(1280, 720)) {
        std::cerr << "Failed to initialize viewer" << std::endl;
        return 1;
    }
    
    viewer.UpdateGaussians(gaussians);
    viewer.Run();
    
    std::cout << "Viewer closed. Exiting..." << std::endl;
    
    return 0;
}
