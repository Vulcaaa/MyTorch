#include "Loss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

// -log(softmax(z)_k) = log(sum_j exp(z_j)) - z_k. The largest logit m is factored out
// so exp never overflows: log(sum_j exp(z_j)) - z_k = (m - z_k) + log(sum_j exp(z_j - m)).
double crossEntropy(const std::vector<double>& logits, const std::vector<double>& target)
{
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double total = 0.0;
    for (const double z : logits)
        total += std::exp(z - maximum);
    const double logTotal = std::log(total);

    double loss = 0.0;
    for (std::size_t k = 0; k < logits.size(); ++k)
        loss += target[k] * ((maximum - logits[k]) + logTotal);
    return loss;
}

// -[y log(p) + (1 - y) log(1 - p)] with p = sigmoid(z) rewritten as
// max(z, 0) - z y + log(1 + exp(-|z|)), where exp never overflows.
double binaryCrossEntropy(const std::vector<double>& logits, const std::vector<double>& target)
{
    double loss = 0.0;
    for (std::size_t k = 0; k < logits.size(); ++k) {
        const double z = logits[k];
        loss += std::max(z, 0.0) - z * target[k] + std::log1p(std::exp(-std::abs(z)));
    }
    return loss;
}

} // namespace

double sampleLoss(const std::vector<double>& logits, const std::vector<double>& target,
                  const std::string& outputActivation)
{
    if (logits.empty() || logits.size() != target.size())
        throw std::invalid_argument("loss requires one target value per output");
    if (outputActivation == "softmax")
        return crossEntropy(logits, target);
    if (outputActivation == "sigmoid")
        return binaryCrossEntropy(logits, target);
    throw std::invalid_argument("no loss is defined for output activation '"
        + outputActivation + "'");
}
