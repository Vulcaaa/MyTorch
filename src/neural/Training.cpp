#include "NeuralNetwork.hpp"
#include "Activation.hpp"
#include "Loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace {

// Zero-filled values shaped like the given parameters.
std::vector<Layer> zerosLike(const std::vector<Layer>& layers)
{
    std::vector<Layer> zeros;
    zeros.reserve(layers.size());
    for (const auto& layer : layers)
        zeros.emplace_back(layer.weights.front().size(), layer.weights.size());
    return zeros;
}

template <typename Function>
void forEachValue(std::vector<Layer>& layers, Function function)
{
    for (auto& layer : layers) {
        for (auto& row : layer.weights)
            for (auto& value : row)
                function(value);
        for (auto& value : layer.biases)
            function(value);
    }
}

} // namespace

double NeuralNetwork::trainBatch(const std::vector<TrainingSample>& batch)
{
    return trainBatchView({batch, nullptr, batch.size()});
}

double NeuralNetwork::trainBatch(const std::vector<TrainingSample>& samples,
    const std::vector<std::size_t>& order, std::size_t begin, std::size_t count)
{
    if (count == 0 || begin > order.size() || count > order.size() - begin)
        throw std::invalid_argument("invalid or empty training batch index range");
    for (std::size_t i = begin; i < begin + count; ++i)
        if (order[i] >= samples.size())
            throw std::invalid_argument("training batch sample index out of range");
    return trainBatchView({samples, order.data() + begin, count});
}

double NeuralNetwork::trainBatchView(SampleView batch)
{
    const double rate = _config.learning_rate;
    if (!std::isfinite(rate) || rate <= 0.0)
        throw std::invalid_argument("learning_rate must be finite and positive");
    auto [batchLoss, updated] = lossAndGradients(batch);

    // Gradient descent: every parameter takes a step of size rate against its gradient.
    // The new values overwrite the gradients and are kept only if they are all finite.
    bool finite = true;
    const auto step = [&](double parameter, double& gradient) {
        gradient = parameter - rate * gradient;
        finite = finite && std::isfinite(gradient);
    };
    for (std::size_t i = 0; i < _layers.size(); ++i) {
        for (std::size_t j = 0; j < _layers[i].weights.size(); ++j) {
            for (std::size_t k = 0; k < _layers[i].weights[j].size(); ++k)
                step(_layers[i].weights[j][k], updated[i].weights[j][k]);
            step(_layers[i].biases[j], updated[i].biases[j]);
        }
    }
    if (!finite)
        throw std::runtime_error("parameters are not finite after the update "
            "(try a lower learning_rate)");
    _layers = std::move(updated);
    return batchLoss;
}

double NeuralNetwork::loss(const std::vector<TrainingSample>& samples) const
{
    validateSamples(samples);
    double total = 0.0;
    ForwardPass pass(_layers);
    for (const auto& sample : samples) {
        forward(sample.inputs, pass);
        total += sampleLoss(pass.outputLogits, sample.target, _config.output_activation);
    }

    const double mean = total / static_cast<double>(samples.size());
    if (!std::isfinite(mean))
        throw std::runtime_error("loss is not finite: network outputs overflowed "
            "(try a lower learning_rate)");
    return mean;
}

std::vector<Layer> NeuralNetwork::gradients(const std::vector<TrainingSample>& samples) const
{
    return lossAndGradients({samples, nullptr, samples.size()}).gradients;
}

NeuralNetwork::LossGradients NeuralNetwork::lossAndGradients(
    SampleView samples) const
{
    validateSamples(samples);
    LossGradients result{0.0, zerosLike(_layers)};
    ForwardPass pass(_layers);
    std::vector<std::vector<double>> deltas;
    deltas.reserve(_layers.size());
    for (const auto& layer : _layers)
        deltas.emplace_back(layer.biases.size());
    for (std::size_t i = 0; i < samples.count; ++i)
        result.loss += backpropagate(samples[i], result.gradients, pass, deltas);

    const double count = static_cast<double>(samples.count);
    result.loss /= count;
    bool finite = std::isfinite(result.loss);
    forEachValue(result.gradients, [&](double& value) {
        value /= count;
        finite = finite && std::isfinite(value);
    });
    if (!finite)
        throw std::runtime_error("loss or gradients are not finite: network outputs overflowed "
            "(try a lower learning_rate)");
    return result;
}

double NeuralNetwork::backpropagate(const TrainingSample& sample, std::vector<Layer>& sums,
    ForwardPass& pass, std::vector<std::vector<double>>& deltas) const
{
    forward(sample.inputs, pass);
    const auto& activations = pass.activations;

    // Output delta (loss derivative with respect to the output logits): p - y, both for
    // softmax with cross-entropy and for sigmoid with binary cross-entropy.
    auto& outputDelta = deltas.back();
    for (std::size_t j = 0; j < outputDelta.size(); ++j)
        outputDelta[j] = activations.back()[j] - sample.target[j];

    // Walk the layers backwards, using the original sample as the first layer input.
    for (std::size_t i = _layers.size(); i > 0; --i) {
        const auto& layer = _layers[i - 1];
        const auto& inputs = i == 1 ? sample.inputs : activations[i - 2];
        const auto& delta = deltas[i - 1];
        const auto& nonzero = pass.nonzero[i - 1];
        const bool sparse = nonzero.size() < inputs.size() / 2;
        auto& sum = sums[i - 1];

        // Weight gradient: delta[j] * inputs[k]. Bias gradient: delta[j].
        for (std::size_t j = 0; j < delta.size(); ++j) {
            sum.biases[j] += delta[j];
            if (delta[j] == 0.0)
                continue;
            if (sparse) {
                for (const auto k : nonzero)
                    sum.weights[j][k] += delta[j] * inputs[k];
            } else {
                for (std::size_t k = 0; k < inputs.size(); ++k)
                    sum.weights[j][k] += delta[j] * inputs[k];
            }
        }
        if (i == 1)
            break;

        // Delta of the previous (hidden) layer: weights^T * delta, times f'(its outputs).
        auto& previous = deltas[i - 2];
        std::fill(previous.begin(), previous.end(), 0.0);
        for (std::size_t j = 0; j < delta.size(); ++j) {
            if (delta[j] == 0.0)
                continue;
            for (std::size_t k = 0; k < inputs.size(); ++k)
                previous[k] += layer.weights[j][k] * delta[j];
        }
        applyActivationDerivative(previous, inputs, _config.hidden_activation);
    }
    return sampleLoss(pass.outputLogits, sample.target, _config.output_activation);
}
