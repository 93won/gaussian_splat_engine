cmake_minimum_required(VERSION 3.18)
project(GaussianSplatEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find packages
find_package(Eigen3 REQUIRED)
find_package(GLEW REQUIRED)
find_package(OpenCV REQUIRED)

# Build Pangolin from thirdparty
set(BUILD_PANGOLIN_EXAMPLES OFF CACHE BOOL "Build Pangolin examples" FORCE)
set(BUILD_PANGOLIN_TESTS OFF CACHE BOOL "Build Pangolin tests" FORCE)
set(BUILD_PANGOLIN_TOOLS OFF CACHE BOOL "Build Pangolin tools" FORCE)
set(BUILD_PANGOLIN_PYTHON OFF CACHE BOOL "Build Pangolin Python" FORCE)
set(BUILD_PYPANGOLIN OFF CACHE BOOL "Build PyPangolin" FORCE)
add_subdirectory(thirdparty/pangolin EXCLUDE_FROM_ALL)

# Include directories
include_directories(src)
include_directories(src/database)
include_directories(src/renderer)
include_directories(src/util)
include_directories(src/viewer)
include_directories(app)
include_directories(thirdparty/pangolin/include)
include_directories(${CMAKE_BINARY_DIR}/thirdparty/pangolin/include)

# Core library
add_library(gaussian_splat_engine STATIC
    src/database/GaussianModel.cpp
    src/database/Camera.cpp
    src/renderer/GaussianRasterizer.cpp
    src/util/PLYUtils.cpp
    src/util/JSONUtils.cpp
)

target_link_libraries(gaussian_splat_engine
    Eigen3::Eigen
    ${OpenCV_LIBS}
    pango_display
    pango_plot
    pango_opengl
    pango_windowing
    pango_vars
    pango_core
    GLEW::GLEW
    GL
)

# Test GaussianRasterizer with 3D visualization (main application)
add_executable(test_rasterizer_3d
    app/test_render_model_3d.cpp
)

target_link_libraries(test_rasterizer_3d
    gaussian_splat_engine
    Eigen3::Eigen
    ${OpenCV_LIBS}
    pango_display
    pango_plot
    pango_opengl
    pango_windowing
    pango_vars
    pango_core
    GLEW::GLEW
    GL
)

target_compile_options(test_rasterizer_3d PRIVATE -Wall -Wextra -O3 -fopenmp)
target_link_libraries(test_rasterizer_3d -fopenmp)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
