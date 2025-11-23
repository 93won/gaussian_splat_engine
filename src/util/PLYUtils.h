/**
 * @file      PLYUtils.h
 * @brief     PLY file I/O utilities for Gaussian Model
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 */

#pragma once

#include <string>
#include <vector>
#include "../database/GaussianModel.h"

namespace ply {

/**
 * @brief Load Gaussians from binary PLY file
 * @param ply_path Path to PLY file
 * @param gaussians Output vector of Gaussians
 * @return true if successful, false otherwise
 */
bool LoadGaussians(const std::string& ply_path, std::vector<Gaussian>& gaussians);

/**
 * @brief Save Gaussians to binary PLY file
 * @param ply_path Path to output PLY file
 * @param gaussians Vector of Gaussians to save
 * @return true if successful, false otherwise
 */
bool SaveGaussians(const std::string& ply_path, const std::vector<Gaussian>& gaussians);

} // namespace ply
