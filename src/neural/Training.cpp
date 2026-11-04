#include "NeuralNetwork.hpp"
#include "Activation.hpp"
#include "Loss.hpp"

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

// Issue #5 replaces this explicit failure with the gradient descent update.
double NeuralNetwork::trainBatch(const std::vector<TrainingSample>&)
{
    throw std::runtime_error("trainBatch is not implemented yet (issue #5)");
}

double NeuralNetwork::loss(const std::vector<TrainingSample>& samples) const
{
    validateSamples(samples);
    double total = 0.0;
    for (const auto& sample : samples)
        total += sampleLoss(forward(sample.inputs).outputLogits, sample.target,
            _config.output_activation);

    const double mean = total / static_cast<double>(samples.size());
    if (!std::isfinite(mean))
        throw std::runtime_error("loss is not finite: network outputs overflowed "
            "(try a lower learning_rate)");
    return mean;
}

std::vector<Layer> NeuralNetwork::gradients(const std::vector<TrainingSample>& samples) const
{
    return lossAndGradients(samples).gradients;
}

NeuralNetwork::LossGradients NeuralNetwork::lossAndGradients(
    const std::vector<TrainingSample>& samples) const
{
    validateSamples(samples);
    LossGradients result{0.0, zerosLike(_layers)};
    for (const auto& sample : samples)
        result.loss += backpropagate(sample, result.gradients);

    const double count = static_cast<double>(samples.size());
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

double NeuralNetwork::backpropagate(const TrainingSample& sample, std::vector<Layer>& sums) const
{
    const auto pass = forward(sample.inputs);
    const auto& activations = pass.activations;

    // Output delta (loss derivative with respect to the output logits): p - y, both for
    // softmax with cross-entropy and for sigmoid with binary cross-entropy.
    std::vector<double> delta = activations.back();
    for (std::size_t j = 0; j < delta.size(); ++j)
        delta[j] -= sample.target[j];

    // Walk the layers backwards: _layers[i] turns activations[i] into activations[i + 1].
    for (std::size_t i = _layers.size(); i > 0; --i) {
        const auto& layer = _layers[i - 1];
        const auto& inputs = activations[i - 1];
        auto& sum = sums[i - 1];

        // Weight gradient: delta[j] * inputs[k]. Bias gradient: delta[j].
        for (std::size_t j = 0; j < delta.size(); ++j) {
            sum.biases[j] += delta[j];
            for (std::size_t k = 0; k < inputs.size(); ++k)
                sum.weights[j][k] += delta[j] * inputs[k];
        }
        if (i == 1)
            break;

        // Delta of the previous (hidden) layer: weights^T * delta, times f'(its outputs).
        std::vector<double> previous(inputs.size(), 0.0);
        for (std::size_t j = 0; j < delta.size(); ++j)
            for (std::size_t k = 0; k < inputs.size(); ++k)
                previous[k] += layer.weights[j][k] * delta[j];
        const auto derivatives = activationDerivative(inputs, _config.hidden_activation);
        for (std::size_t k = 0; k < previous.size(); ++k)
            previous[k] *= derivatives[k];
        delta = std::move(previous);
    }
    return sampleLoss(pass.outputLogits, sample.target, _config.output_activation);
}
