#pragma once

#include "mytorch/NetworkConfig.hpp"
#include <cstddef>
#include <vector>

struct Layer {
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;

    Layer(std::size_t input_size, std::size_t neuron_count)
        : weights(neuron_count, std::vector<double>(input_size, 0.0)),
          biases(neuron_count, 0.0)
    {}
};

class NeuralNetwork {
    public:
        explicit NeuralNetwork(const NetworkConfig &config);

    private:
        friend class ModelSerializer;

        // Restore validated parameters without random initialization.
        NeuralNetwork(NetworkConfig config, std::vector<Layer> layers);

        NetworkConfig _config;
        std::vector<Layer> _layers;
};
