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
    std::string optimizer = "sgd";
    // Probability of zeroing each hidden activation during training (0 disables dropout).
    double dropout = 0.0;
};

// Throws std::runtime_error when the file cannot be read or is invalid.
NetworkConfig loadNetworkConfig(const std::string &filepath);

// Summarizes a configuration on one line, with the keys of the configuration file:
// "layers=777,128,64,3 hidden_activation=sigmoid output_activation=softmax
// initialization=xavier learning_rate=0.01 seed=42 optimizer=sgd"; " dropout=0.1" follows
// when dropout is enabled.
std::string describeNetworkConfig(const NetworkConfig &config);
