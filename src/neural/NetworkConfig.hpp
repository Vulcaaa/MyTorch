#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct NetworkConfig {
    std::vector<std::size_t> layers;
    std::string hidden_activation;
    std::string output_activation;
    double learning_rate = 0.01;
    std::string initialization;
    std::uint32_t seed = 42;
};

// Throws std::runtime_error when the file cannot be read or is invalid.
NetworkConfig loadNetworkConfig(const std::string &filepath);
