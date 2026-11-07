#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

// The CSV remains the numerical source; this export needs no Python runtime.
void saveLossChart(const std::string& path, const std::vector<std::array<double, 3>>& rows,
    std::size_t bestEpoch, double initialValidationLoss);
