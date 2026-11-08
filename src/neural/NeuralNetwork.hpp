#pragma once

#include "NetworkConfig.hpp"
#include <cstddef>
#include <vector>

// Generic numeric example shared by the training loop and the neural backend.
struct TrainingSample {
    std::vector<double> inputs;
    std::vector<double> target;
    // Optional exact sparse storage of binary inputs; generic dense samples remain valid.
    std::vector<std::size_t> activeInputs{};
    std::size_t binaryInputSize = 0;
    std::vector<double> expandInputs() const
    {
        std::vector<double> values(binaryInputSize, 0.0);
        for (auto index : activeInputs)
            values.at(index) = 1.0;
        return values;
    }
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
        // One mini-batch gradient descent step: every weight and bias moves by
        // -learning_rate * gradient of the mean batch loss. Returns the mean loss before the
        // update. Throws like gradients(), std::invalid_argument when learning_rate is not
        // finite and positive, and std::runtime_error when an updated parameter would not be
        // finite. The network is unchanged whenever it throws.
        double trainBatch(const std::vector<TrainingSample>& batch);
        // Same update using a slice of shuffled sample indices, without copying samples.
        // Rejects empty slices, invalid ranges, and out-of-range sample indices.
        double trainBatch(const std::vector<TrainingSample>& samples,
            const std::vector<std::size_t>& order, std::size_t begin, std::size_t count);
        // Mean loss over samples without changing the network: cross-entropy for softmax,
        // binary cross-entropy summed over outputs for sigmoid. Throws std::invalid_argument
        // for invalid samples (see validateSamples) and std::runtime_error if it is not finite.
        double loss(const std::vector<TrainingSample>& samples) const;
        // Gradients of loss(samples) with respect to every weight and bias, shaped like layers(),
        // computed by backpropagation without changing the network. Throws like loss(), and
        // std::invalid_argument when the hidden activation has no derivative.
        std::vector<Layer> gradients(const std::vector<TrainingSample>& samples) const;
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
            explicit ForwardPass(const std::vector<Layer>& layers);
            // Layer outputs only; the original input vector is borrowed, never copied.
            std::vector<std::vector<double>> activations;
            // Nonzero input indices per layer, in ascending order; reused across samples.
            std::vector<std::vector<std::size_t>> nonzero;
            // Output layer values before its activation, used for a numerically stable loss.
            std::vector<double> outputLogits;
        };

        void forward(const std::vector<double>& inputs, ForwardPass& pass) const;

        struct SampleView {
            const std::vector<TrainingSample>& samples;
            const std::size_t* indices;
            std::size_t count;
            const TrainingSample& operator[](std::size_t index) const
            { return samples[indices ? indices[index] : index]; }
        };
        void validateSamples(SampleView samples) const;
        double trainBatchView(SampleView samples);

        struct LossGradients {
            double loss;
            std::vector<Layer> gradients;
        };

        // Mean loss and gradients over validated samples, from one forward pass per sample.
        LossGradients lossAndGradients(SampleView samples) const;
#ifdef MYTORCH_USE_BLAS
        // Validated samples, processed in bounded contiguous blocks; optional backprop.
        LossGradients batchedLossGradients(SampleView samples, bool gradients) const;
#endif
        // Adds the gradients of one sample's loss to sums and returns that loss.
        double backpropagate(const TrainingSample& sample, std::vector<Layer>& sums,
            ForwardPass& pass, std::vector<std::vector<double>>& deltas) const;

        NetworkConfig _config;
        std::vector<Layer> _layers;
};
