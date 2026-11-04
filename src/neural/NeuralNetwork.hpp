#pragma once

#include "NetworkConfig.hpp"
#include <cstddef>
#include <vector>

// Generic numeric example shared by the training loop and the neural backend.
struct TrainingSample {
    std::vector<double> inputs;
    std::vector<double> target;
};

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
        // Uses the given parameters instead of initializing them.
        // Throws std::invalid_argument when they do not match config.layers or are not finite.
        NeuralNetwork(NetworkConfig config, std::vector<Layer> layers);

        std::vector<double> predict(const std::vector<double>& inputs) const;
        // Implemented by issue #5; currently throw until the backend is available.
        double trainBatch(const std::vector<TrainingSample>& batch);
        double loss(const std::vector<TrainingSample>& samples) const;
        std::size_t inputSize() const { return _config.layers.front(); }
        std::size_t outputSize() const { return _config.layers.back(); }
        const NetworkConfig& config() const { return _config; }
        // Parameter layers: layers()[i] connects config().layers[i] to config().layers[i + 1].
        const std::vector<Layer> &layers() const { return _layers; }

    private:
        friend class ModelSerializer;

        std::vector<std::vector<double>> forward(const std::vector<double> &inputs) const;

        NetworkConfig _config;
        std::vector<Layer> _layers;
};
