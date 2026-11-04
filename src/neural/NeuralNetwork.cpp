#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <iostream>

#include "NeuralNetwork.hpp"
#include "Activation.hpp"
#include "Initializer.hpp"

namespace {

constexpr double targetSumTolerance = 1e-6;

void validateLayerSizes(const NetworkConfig &config)
{
    if (config.layers.size() < 2)
        throw std::invalid_argument("a network requires at least an input and output layer");
    for (const auto size : config.layers)
        if (size == 0)
            throw std::invalid_argument("layer sizes must be positive");
}

void validateParameters(const NetworkConfig &config, const std::vector<Layer> &layers)
{
    if (layers.size() != config.layers.size() - 1)
        throw std::invalid_argument("expected " + std::to_string(config.layers.size() - 1)
            + " parameter layers, got " + std::to_string(layers.size()));
    for (std::size_t i = 0; i < layers.size(); ++i) {
        const auto inputs = config.layers[i];
        const auto neurons = config.layers[i + 1];
        const auto &layer = layers[i];
        const auto where = "parameter layer " + std::to_string(i) + ": ";

        if (layer.weights.size() != neurons || layer.biases.size() != neurons)
            throw std::invalid_argument(where + "expected " + std::to_string(neurons)
                + " weight rows and biases");
        for (const auto &row : layer.weights) {
            if (row.size() != inputs)
                throw std::invalid_argument(where + "expected " + std::to_string(inputs)
                    + " weights per neuron");
            for (const auto weight : row)
                if (!std::isfinite(weight))
                    throw std::invalid_argument(where + "weights must be finite");
        }
        for (const auto bias : layer.biases)
            if (!std::isfinite(bias))
                throw std::invalid_argument(where + "biases must be finite");
    }
}

} // namespace

NeuralNetwork::NeuralNetwork(const NetworkConfig& config) : _config(config)
{
    validateLayerSizes(_config);

    std::mt19937 rng(_config.seed);
    const auto method = initialization::parseMethod(_config.initialization);
    _layers.reserve(config.layers.size() - 1);
    for (std::size_t i = 1; i < config.layers.size(); ++i) {
        _layers.emplace_back(config.layers[i - 1], config.layers[i]);
        initialization::initialize(_layers.back(), method, rng);
    }
}

NeuralNetwork::ForwardPass NeuralNetwork::forward(const std::vector<double> &inputs) const
{
    if (_layers.empty()) {
        std::cerr << "Error: NeuralNetwork requires at least one parameter layer.\n";
        return {};
    }
    ForwardPass pass;
    pass.activations.push_back(inputs);

    for (const auto &layer : _layers) {
        const auto &current_inputs = pass.activations.back();
        std::vector<double> results(layer.weights.size());

        for (std::size_t j = 0; j < layer.weights.size(); ++j) {
            const auto &weights = layer.weights[j];
            if (current_inputs.size() != weights.size()) {
                std::cerr << "Error: Mismatched number of inputs for neuron.\n";
                return {};
            }
            double sum = layer.biases[j];
            for (std::size_t k = 0; k < weights.size(); ++k)
                sum += current_inputs[k] * weights[k];
            results[j] = sum;
        }
        const bool isOutput = (&layer == &_layers.back());
        if (isOutput)
            pass.outputLogits = results;
        applyActivation(results, isOutput ? _config.output_activation : _config.hidden_activation);
        pass.activations.push_back(std::move(results));
    }
    return pass;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) const
{
    auto pass = forward(inputs);

    if (pass.activations.empty())
        throw std::runtime_error("Forward pass failed.");
    return std::move(pass.activations.back());
}

void NeuralNetwork::validateSamples(const std::vector<TrainingSample> &samples) const
{
    const auto &activation = _config.output_activation;
    if (activation != "softmax" && activation != "sigmoid")
        throw std::invalid_argument("no loss is defined for output activation '" + activation + "'");
    if (samples.empty())
        throw std::invalid_argument("training requires at least one sample");

    const auto fail = [](std::size_t index, const std::string &message) {
        throw std::invalid_argument("training sample " + std::to_string(index) + ": " + message);
    };
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto &sample = samples[i];
        if (sample.inputs.size() != inputSize())
            fail(i, "expected " + std::to_string(inputSize()) + " inputs, got "
                + std::to_string(sample.inputs.size()));
        if (sample.target.size() != outputSize())
            fail(i, "expected " + std::to_string(outputSize()) + " target values, got "
                + std::to_string(sample.target.size()));
        for (const auto value : sample.inputs)
            if (!std::isfinite(value))
                fail(i, "inputs must be finite");
        double sum = 0.0;
        for (const auto value : sample.target) {
            if (!std::isfinite(value) || value < 0.0 || value > 1.0)
                fail(i, "target values must be finite and in [0, 1]");
            sum += value;
        }
        if (activation == "softmax" && std::abs(sum - 1.0) > targetSumTolerance)
            fail(i, "softmax target values must sum to 1");
    }
}

NeuralNetwork::NeuralNetwork(NetworkConfig config, std::vector<Layer> layers)
    : _config(std::move(config)), _layers(std::move(layers))
{
    validateLayerSizes(_config);
    validateParameters(_config, _layers);
}
