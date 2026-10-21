#include "NeuralNetwork.hpp"
#include "Initializer.hpp"
#include <stdexcept>
#include <utility>

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

NeuralNetwork::NeuralNetwork(NetworkConfig config, std::vector<Layer> layers)
    : _config(std::move(config)), _layers(std::move(layers))
{}
