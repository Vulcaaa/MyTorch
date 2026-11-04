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
        // Implemented by issue #5; currently throws until the backend is available.
        double trainBatch(const std::vector<TrainingSample>& batch);
        // Mean loss over samples without changing the network: cross-entropy for softmax,
        // binary cross-entropy summed over outputs for sigmoid. Throws std::invalid_argument
        // for invalid samples (see validateSamples) and std::runtime_error if it is not finite.
        double loss(const std::vector<TrainingSample>& samples) const;
        std::size_t inputSize() const { return _config.layers.front(); }
        std::size_t outputSize() const { return _config.layers.back(); }
        const NetworkConfig& config() const { return _config; }
        // Parameter layers: layers()[i] connects config().layers[i] to config().layers[i + 1].
        const std::vector<Layer> &layers() const { return _layers; }

        // Checks samples before training or loss evaluation. Throws std::invalid_argument when
        // samples is empty, a sample has the wrong number of inputs or targets, a value is not
        // finite, a target value is outside [0, 1], softmax targets do not sum to 1 (tolerance
        // 1e-6), or the output activation has no supported loss (softmax and sigmoid only).
        void validateSamples(const std::vector<TrainingSample> &samples) const;

    private:
        friend class ModelSerializer;

        // Values computed by one forward pass, kept for loss and backpropagation.
        struct ForwardPass {
            // activations[0] holds the inputs, activations[i + 1] the outputs of _layers[i].
            std::vector<std::vector<double>> activations;
            // Output layer values before its activation, used for a numerically stable loss.
            std::vector<double> outputLogits;
        };

        // Returns an empty pass (no activations) when the inputs do not match the network.
        ForwardPass forward(const std::vector<double> &inputs) const;

        NetworkConfig _config;
        std::vector<Layer> _layers;
};
