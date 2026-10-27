#include <stdexcept>
#include <utility>
#include <iostream>

#include "NeuralNetwork.hpp"
#include "Activation.hpp"
#include "Initializer.hpp"

NeuralNetwork::NeuralNetwork(const NetworkConfig& config) : _config(config)
{
    if (config.layers.size() < 2)
        throw std::invalid_argument("a network requires at least an input and output layer");
    for (const auto size : config.layers)
        if (size == 0)
            throw std::invalid_argument("layer sizes must be positive");

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
{}
