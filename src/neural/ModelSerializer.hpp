#pragma once

#include "NeuralNetwork.hpp"
#include <string>

class ModelSerializer {
    public:
        // Throws std::runtime_error on invalid model data or file errors.
        static void save(const NeuralNetwork &network, const std::string &filepath);
        static NeuralNetwork load(const std::string &filepath);
};
