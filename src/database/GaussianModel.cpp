/**
 * @file      GaussianModel.cpp
 * @brief     Gaussian primitive data structure and model implementation
 * @author    Seungwon Choi
 * @email     csw3575@snu.ac.kr
 * @date      2025-11-23
 * 
 */

#include "GaussianModel.h"
#include "../util/PLYUtils.h"

bool GaussianModel::LoadFromPLY(const std::string& ply_path) {
    return ply::LoadGaussians(ply_path, m_gaussians);
}

bool GaussianModel::SaveToPLY(const std::string& ply_path) const {
    return ply::SaveGaussians(ply_path, m_gaussians);
}
