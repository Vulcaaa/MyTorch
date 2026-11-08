#include "NeuralNetwork.hpp"

#ifdef MYTORCH_USE_BLAS

#include "Activation.hpp"
#include "Loss.hpp"

#ifdef MYTORCH_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

// Bound temporary memory even when loss()/gradients() receive the entire dataset.
constexpr std::size_t blockSize = 256;

int blasSize(std::size_t size)
{
    if (size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("network dimension exceeds the BLAS integer range");
    return static_cast<int>(size);
}

} // namespace

NeuralNetwork::LossGradients NeuralNetwork::batchedLossGradients(
    SampleView samples, bool gradients) const
{
    const auto capacity = std::min(blockSize, samples.count);
    std::vector<std::vector<double>> weights, activations, deltas, weightGradients;
    std::vector<std::vector<std::size_t>> nonzero(capacity);
    activations.emplace_back(capacity * inputSize());
    for (std::size_t i = 0; i < _layers.size(); ++i) {
        const auto& layer = _layers[i];
        const auto inputs = layer.weights.front().size();
        const auto outputs = layer.biases.size();
        blasSize(inputs);
        blasSize(outputs);
        weights.emplace_back();
        // Sparse chess inputs do not need a dense copy of the largest weight matrix.
        if (i != 0) {
            weights.back().reserve(inputs * outputs);
            for (const auto& row : layer.weights)
                weights.back().insert(weights.back().end(), row.begin(), row.end());
        }
        activations.emplace_back(capacity * outputs);
        if (gradients) {
            deltas.emplace_back(capacity * outputs);
            weightGradients.emplace_back(i == 0 ? 0 : inputs * outputs, 0.0);
        }
    }
    // Dropout scales per layer output (f'(activation) times mask), only while training.
    const bool dropout = gradients && _config.dropout > 0.0;
    std::vector<std::vector<double>> dropoutScale(_layers.size() + 1);
    LossGradients result{0.0, {}};
    if (gradients)
        for (const auto& layer : _layers)
            result.gradients.emplace_back(layer.weights.front().size(), layer.biases.size());

    for (std::size_t begin = 0; begin < samples.count;) {
        const auto count = std::min(capacity, samples.count - begin);
        const int rows = blasSize(count);
        std::size_t nonzeroCount = 0;
        for (std::size_t s = 0; s < count; ++s) {
            const auto& sample = samples[begin + s];
            auto* inputs = activations.front().data() + s * inputSize();
            nonzero[s].clear();
            if (sample.binaryInputSize) {
                std::fill(inputs, inputs + inputSize(), 0.0);
                nonzero[s] = sample.activeInputs;
                for (auto k : nonzero[s])
                    inputs[k] = 1.0;
            } else {
                std::copy(sample.inputs.begin(), sample.inputs.end(), inputs);
                for (std::size_t k = 0; k < inputSize(); ++k)
                    if (inputs[k] != 0.0)
                        nonzero[s].push_back(k);
            }
            nonzeroCount += nonzero[s].size();
        }
        const bool sparseInput = nonzeroCount < count * inputSize() / 2;

        for (std::size_t i = 0; i < _layers.size(); ++i) {
            const auto& layer = _layers[i];
            const auto inputs = layer.weights.front().size();
            const auto outputs = layer.biases.size();
            const bool output = i + 1 == _layers.size();
            if (i == 0 && sparseInput) {
                for (std::size_t s = 0; s < count; ++s)
                    for (std::size_t j = 0; j < outputs; ++j) {
                        double sum = 0.0;
                        for (const auto k : nonzero[s])
                            sum += activations[0][s * inputs + k] * layer.weights[j][k];
                        activations[1][s * outputs + j] = sum;
                    }
            } else {
                if (weights[i].empty()) {
                    weights[i].reserve(inputs * outputs);
                    for (const auto& row : layer.weights)
                        weights[i].insert(weights[i].end(), row.begin(), row.end());
                }
                // Z[B,O] = A[B,I] * W[O,I]^T. BLAS owns matrix parallelism.
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    rows, blasSize(outputs), blasSize(inputs), 1.0,
                    activations[i].data(), blasSize(inputs), weights[i].data(), blasSize(inputs),
                    0.0, activations[i + 1].data(), blasSize(outputs));
            }
            for (std::size_t s = 0; s < count; ++s) {
                auto* row = activations[i + 1].data() + s * outputs;
                for (std::size_t j = 0; j < outputs; ++j)
                    row[j] += layer.biases[j];
                if (output)
                    result.loss += sampleLoss(row, samples[begin + s].target.data(),
                        outputs, _config.output_activation);
                if (!output || gradients)
                    applyActivation(row, outputs,
                        output ? _config.output_activation : _config.hidden_activation);
                if (dropout && !output) {
                    dropoutScale[i + 1].resize(capacity * outputs);
                    applyDropout(row, dropoutScale[i + 1].data() + s * outputs, outputs);
                }
            }
        }

        if (gradients) {
            for (std::size_t s = 0; s < count; ++s)
                for (std::size_t j = 0; j < outputSize(); ++j)
                    deltas.back()[s * outputSize() + j] =
                        activations.back()[s * outputSize() + j] - samples[begin + s].target[j];

            for (std::size_t remaining = _layers.size(); remaining > 0; --remaining) {
                const auto i = remaining - 1;
                const auto inputs = _layers[i].weights.front().size();
                const auto outputs = _layers[i].biases.size();
                // Sum across blocks; division happens once, over the whole requested batch.
                // dW[O,I] += D[B,O]^T * A[B,I].
                if (i == 0 && sparseInput) {
                    for (std::size_t s = 0; s < count; ++s)
                        for (std::size_t j = 0; j < outputs; ++j) {
                            const auto delta = deltas[i][s * outputs + j];
                            if (delta != 0.0)
                                for (const auto k : nonzero[s])
                                    result.gradients[i].weights[j][k] += delta * activations[0][s * inputs + k];
                        }
                } else {
                    if (weightGradients[i].empty())
                        weightGradients[i].resize(inputs * outputs, 0.0);
                    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        blasSize(outputs), blasSize(inputs), rows, 1.0,
                        deltas[i].data(), blasSize(outputs), activations[i].data(), blasSize(inputs),
                        1.0, weightGradients[i].data(), blasSize(inputs));
                }
                for (std::size_t s = 0; s < count; ++s)
                    for (std::size_t j = 0; j < outputs; ++j)
                        result.gradients[i].biases[j] += deltas[i][s * outputs + j];
                if (i == 0)
                    break;
                // Dprev[B,I] = D[B,O] * W[O,I], then multiply by f'(Aprev).
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    rows, blasSize(inputs), blasSize(outputs), 1.0,
                    deltas[i].data(), blasSize(outputs), weights[i].data(), blasSize(inputs),
                    0.0, deltas[i - 1].data(), blasSize(inputs));
                if (dropout) {
                    for (std::size_t k = 0; k < count * inputs; ++k)
                        deltas[i - 1][k] *= dropoutScale[i][k];
                } else {
                    applyActivationDerivative(deltas[i - 1].data(), activations[i].data(),
                        count * inputs, _config.hidden_activation);
                }
            }
        }
        begin += count;
    }

    result.loss /= static_cast<double>(samples.count);
    bool finite = std::isfinite(result.loss);
    for (std::size_t i = 0; i < result.gradients.size(); ++i) {
        auto& layer = result.gradients[i];
        const auto inputs = layer.weights.front().size();
        for (std::size_t j = 0; j < layer.biases.size(); ++j) {
            for (std::size_t k = 0; k < inputs; ++k) {
                const double dense = weightGradients[i].empty() ? 0.0
                    : weightGradients[i][j * inputs + k];
                const double value = (layer.weights[j][k] + dense)
                    / static_cast<double>(samples.count);
                layer.weights[j][k] = value;
                finite = finite && std::isfinite(value);
            }
            layer.biases[j] /= static_cast<double>(samples.count);
            finite = finite && std::isfinite(layer.biases[j]);
        }
    }
    if (!finite)
        throw std::runtime_error("loss or gradients are not finite: network outputs overflowed "
            "(try a lower learning_rate)");
    return result;
}

#endif
