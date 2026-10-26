#include "Activation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

double sigmoid(double value)
{
    if (value >= 0.0)
        return 1.0 / (1.0 + std::exp(-value));
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

void softmax(std::vector<double>& values)
{
    if (values.empty())
        return;

    const double maximum = *std::max_element(values.begin(), values.end());
    double total = 0.0;
    for (double& value : values) {
        value = std::exp(value - maximum);
        total += value;
    }
    for (double& value : values)
        value /= total;
}

} // namespace

void applyActivation(std::vector<double>& values,
                     const std::string& activation)
{
    if (activation == "softmax") {
        softmax(values);
        return;
    }
    if (activation == "linear")
        return;
    if (activation != "relu" && activation != "sigmoid" && activation != "tanh")
        throw std::invalid_argument("Unknown activation: " + activation);

    for (double& value : values) {
        if (activation == "relu")
            value = std::max(0.0, value);
        else if (activation == "sigmoid")
            value = sigmoid(value);
        else
            value = std::tanh(value);
    }
}
