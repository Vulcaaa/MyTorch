#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <iostream>

#include "NeuralNetwork.hpp"
#include "Activation.hpp"
#include "Initializer.hpp"

namespace {

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

std::vector<std::vector<double>> NeuralNetwork::forward(
    const std::vector<double> &inputs) const
{
    if (_layers.empty()) {
        std::cerr << "Error: NeuralNetwork requires at least one parameter layer.\n";
        return {};
    }
    std::vector<std::vector<double>> activations{inputs};

    for (const auto &layer : _layers) {
        const auto &current_inputs = activations.back();
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
        applyActivation(results, isOutput ? _config.output_activation : _config.hidden_activation);
        activations.push_back(std::move(results));
    }
    return activations;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) const
{
    const auto activations = forward(inputs);

    if (activations.empty())
        throw std::runtime_error("Forward pass failed.");
    return activations.back();
}

NeuralNetwork::NeuralNetwork(NetworkConfig config, std::vector<Layer> layers)
    : _config(std::move(config)), _layers(std::move(layers))
{
    validateLayerSizes(_config);
    validateParameters(_config, _layers);
}
