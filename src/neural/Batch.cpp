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
    activations.emplace_back(capacity * inputSize());
    for (const auto& layer : _layers) {
        const auto inputs = layer.weights.front().size();
        const auto outputs = layer.biases.size();
        blasSize(inputs);
        blasSize(outputs);
        weights.emplace_back();
        weights.back().reserve(inputs * outputs);
        for (const auto& row : layer.weights)
            weights.back().insert(weights.back().end(), row.begin(), row.end());
        activations.emplace_back(capacity * outputs);
        if (gradients) {
            deltas.emplace_back(capacity * outputs);
            weightGradients.emplace_back(inputs * outputs, 0.0);
        }
    }
    LossGradients result{0.0, {}};
    if (gradients)
        for (const auto& layer : _layers)
            result.gradients.emplace_back(layer.weights.front().size(), layer.biases.size());

    for (std::size_t begin = 0; begin < samples.count;) {
        const auto count = std::min(capacity, samples.count - begin);
        const int rows = blasSize(count);
        for (std::size_t s = 0; s < count; ++s)
            std::copy(samples[begin + s].inputs.begin(), samples[begin + s].inputs.end(),
                activations.front().data() + s * inputSize());

        for (std::size_t i = 0; i < _layers.size(); ++i) {
            const auto& layer = _layers[i];
            const auto inputs = layer.weights.front().size();
            const auto outputs = layer.biases.size();
            const bool output = i + 1 == _layers.size();
            // Z[B,O] = A[B,I] * W[O,I]^T. BLAS owns matrix parallelism.
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                rows, blasSize(outputs), blasSize(inputs), 1.0,
                activations[i].data(), blasSize(inputs), weights[i].data(), blasSize(inputs),
                0.0, activations[i + 1].data(), blasSize(outputs));
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
                cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    blasSize(outputs), blasSize(inputs), rows, 1.0,
                    deltas[i].data(), blasSize(outputs), activations[i].data(), blasSize(inputs),
                    1.0, weightGradients[i].data(), blasSize(inputs));
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
                applyActivationDerivative(deltas[i - 1].data(), activations[i].data(),
                    count * inputs, _config.hidden_activation);
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
                const double value = weightGradients[i][j * inputs + k]
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
