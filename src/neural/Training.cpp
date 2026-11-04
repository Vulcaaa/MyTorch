#include "NeuralNetwork.hpp"
#include "Loss.hpp"

#include <cmath>
#include <stdexcept>

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
