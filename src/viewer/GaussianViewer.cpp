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
    
    // Define Projection and initial ModelView matrix (store as member)
    m_s_cam = std::make_unique<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(width, height, 500, 500, width/2, height/2, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -2, -5, 0, 0, 0, pangolin::AxisY)
    );
    
    // Create Interactive View in window with name "cam"
    pangolin::View& d_cam = pangolin::Display("cam")
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(250), 1.0, -(float)width/height)
        .SetHandler(new pangolin::Handler3D(*m_s_cam));
    
    m_initialized = true;
    
    std::cout << "[GaussianViewer] Initialized (" << width << "x" << height << ")" << std::endl;
    return true;
}

void GaussianViewer::UpdateGaussians(const std::vector<Gaussian>& gaussians) {
    m_gaussians = gaussians;
    ComputeScaleRange();
    std::cout << "[GaussianViewer] Updated " << m_gaussians.size() << " Gaussians" << std::endl;
    std::cout << "[GaussianViewer] Scale range: [" << m_min_scale << ", " << m_max_scale << "]" << std::endl;
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
        opacity = std::max(0.0f, std::min(1.0f, opacity));
        
        // Color DC with sigmoid
        float r = 1.0f / (1.0f + std::exp(-g.color_dc.x()));
        float g_val = 1.0f / (1.0f + std::exp(-g.color_dc.y()));
        float b = 1.0f / (1.0f + std::exp(-g.color_dc.z()));
        
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
    
    // Create UI panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(250));
    pangolin::Var<float> gaussian_scale("ui.Gaussian Scale", 1.0f, 0.001f, 1.0f);
    pangolin::Var<bool> show_coordinate_frame("ui.Show Axes", true, true);
    pangolin::Var<float> point_size("ui.Point Size", 3.0f, 1.0f, 10.0f);
    pangolin::Var<bool> apply_sigmoid("ui.Apply Sigmoid to Color", false, true);
    
    std::cout << "[GaussianViewer] Starting render loop..." << std::endl;
    std::cout << "[GaussianViewer] Controls:" << std::endl;
    std::cout << "  - Drag to rotate" << std::endl;
    std::cout << "  - Right-click drag to pan" << std::endl;
    std::cout << "  - Scroll to zoom" << std::endl;
    std::cout << "  - 'Gaussian Scale' (0.001-1.0): multiply gaussian size" << std::endl;
    std::cout << "  - 'Point Size': base rendering point size" << std::endl;
    std::cout << "[GaussianViewer] Close window to exit" << std::endl;
    
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Activate camera view
        pangolin::Display("cam").Activate(*m_s_cam);
        
        // Draw coordinate frame
        if (show_coordinate_frame) {
            DrawCoordinateFrame();
        }
        
        // Enable point sprite for variable size points
        glEnable(GL_POINT_SPRITE);
        glEnable(GL_PROGRAM_POINT_SIZE);
        
        // Draw Gaussians - gaussian_scale will be used inside DrawGaussians
        DrawGaussians();
        
        glDisable(GL_PROGRAM_POINT_SIZE);
        glDisable(GL_POINT_SPRITE);
        
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
