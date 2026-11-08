#include <algorithm>
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

void validateDropout(const NetworkConfig& config)
{
    if (!std::isfinite(config.dropout) || config.dropout < 0.0 || config.dropout >= 1.0)
        throw std::invalid_argument("dropout must be in [0, 1)");
}

} // namespace

void NeuralNetwork::setLearningRate(double rate)
{
    if (!std::isfinite(rate) || rate <= 0.0)
        throw std::invalid_argument("learning_rate must be finite and positive");
    _config.learning_rate = rate;
}

void NeuralNetwork::applyDropout(double* activated, double* scale, std::size_t count) const
{
    std::fill(scale, scale + count, 1.0);
    applyActivationDerivative(scale, activated, count, _config.hidden_activation);
    std::bernoulli_distribution keep(1.0 - _config.dropout);
    const double factor = 1.0 / (1.0 - _config.dropout);
    for (std::size_t k = 0; k < count; ++k) {
        const double mask = keep(_dropoutRandom) ? factor : 0.0;
        activated[k] *= mask;
        scale[k] *= mask;
    }
}

NeuralNetwork::NeuralNetwork(const NetworkConfig& config) : _config(config)
{
    validateLayerSizes(_config);
    validateDropout(_config);
    _dropoutRandom.seed(_config.seed);

    std::mt19937 rng(_config.seed);
    const auto method = initialization::parseMethod(_config.initialization);
    _layers.reserve(config.layers.size() - 1);
    for (std::size_t i = 1; i < config.layers.size(); ++i) {
        _layers.emplace_back(config.layers[i - 1], config.layers[i]);
        initialization::initialize(_layers.back(), method, rng);
    }
}

NeuralNetwork::ForwardPass::ForwardPass(const std::vector<Layer>& layers)
{
    activations.reserve(layers.size());
    nonzero.resize(layers.size());
    dropoutScale.resize(layers.size());
    for (std::size_t i = 0; i < layers.size(); ++i) {
        activations.emplace_back(layers[i].biases.size());
        nonzero[i].reserve(layers[i].weights.front().size());
    }
    outputLogits.resize(layers.back().biases.size());
}

void NeuralNetwork::forward(const std::vector<double>& inputs, ForwardPass& pass, bool training) const
{
    if (inputs.size() != inputSize())
        throw std::runtime_error("Forward pass failed: mismatched number of inputs.");

    for (std::size_t i = 0; i < _layers.size(); ++i) {
        const auto& layer = _layers[i];
        const auto& current = i == 0 ? inputs : pass.activations[i - 1];
        auto& results = pass.activations[i];
        auto& nonzero = pass.nonzero[i];
        nonzero.clear();
        for (std::size_t k = 0; k < current.size(); ++k)
            if (current[k] != 0.0)
                nonzero.push_back(k);
        // Sparse inputs (including ReLU hidden outputs) use the same sum order.
        // Dense layers keep a contiguous loop rather than indexed accesses.
        const bool sparse = nonzero.size() < current.size() / 2;
        for (std::size_t j = 0; j < layer.weights.size(); ++j) {
            const auto& weights = layer.weights[j];
            double sum = layer.biases[j];
            if (sparse) {
                for (const auto k : nonzero)
                    sum += current[k] * weights[k];
            } else {
                for (std::size_t k = 0; k < current.size(); ++k)
                    sum += current[k] * weights[k];
            }
            results[j] = sum;
        }
        const bool isOutput = i + 1 == _layers.size();
        if (isOutput)
            std::copy(results.begin(), results.end(), pass.outputLogits.begin());
        applyActivation(results, isOutput ? _config.output_activation : _config.hidden_activation);
        if (training && !isOutput && _config.dropout > 0.0) {
            pass.dropoutScale[i].resize(results.size());
            applyDropout(results.data(), pass.dropoutScale[i].data(), results.size());
        }
    }
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& inputs) const
{
    ForwardPass pass(_layers);
    forward(inputs, pass);
    return std::move(pass.activations.back());
}

void NeuralNetwork::validateSamples(const std::vector<TrainingSample> &samples) const
{
    validateSamples(SampleView{samples, nullptr, samples.size()});
}

void NeuralNetwork::validateSamples(SampleView samples) const
{
    const auto &activation = _config.output_activation;
    if (activation != "softmax" && activation != "sigmoid")
        throw std::invalid_argument("no loss is defined for output activation '" + activation + "'");
    if (samples.count == 0)
        throw std::invalid_argument("training requires at least one sample");

    const auto fail = [](std::size_t index, const std::string &message) {
        throw std::invalid_argument("training sample " + std::to_string(index) + ": " + message);
    };
    for (std::size_t i = 0; i < samples.count; ++i) {
        const auto &sample = samples[i];
        const auto width = sample.binaryInputSize ? sample.binaryInputSize : sample.inputs.size();
        if (width != inputSize())
            fail(i, "expected " + std::to_string(inputSize()) + " inputs, got "
                + std::to_string(width));
        if (sample.binaryInputSize) {
            if (!sample.inputs.empty())
                fail(i, "binary samples must not also contain dense inputs");
            std::size_t previous = 0;
            bool first = true;
            for (auto index : sample.activeInputs) {
                if (index >= width || (!first && index <= previous))
                    fail(i, "binary input indices must be in range and strictly increasing");
                first = false;
                previous = index;
            }
        } else if (!sample.activeInputs.empty()) {
            fail(i, "binary indices require an input size");
        }
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
    validateDropout(_config);
    _dropoutRandom.seed(_config.seed);
    validateParameters(_config, _layers);
}
