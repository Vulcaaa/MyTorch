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

std::vector<double> activationDerivative(const std::vector<double>& activated,
                                         const std::string& activation)
{
    std::vector<double> derivatives(activated.size());
    const auto apply = [&](auto derivative) {
        std::transform(activated.begin(), activated.end(), derivatives.begin(), derivative);
    };

    if (activation == "sigmoid")
        apply([](double a) { return a * (1.0 - a); });
    else if (activation == "tanh")
        apply([](double a) { return 1.0 - a * a; });
    else if (activation == "relu")
        apply([](double a) { return a > 0.0 ? 1.0 : 0.0; });
    else if (activation == "linear")
        std::fill(derivatives.begin(), derivatives.end(), 1.0);
    else if (activation == "softmax")
        throw std::invalid_argument("softmax has no element-wise derivative; "
            "its gradient is computed with the cross-entropy loss");
    else
        throw std::invalid_argument("Unknown activation: " + activation);
    return derivatives;
}
