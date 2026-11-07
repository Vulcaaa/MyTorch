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

void softmax(double* values, std::size_t count)
{
    if (count == 0)
        return;

    const double maximum = *std::max_element(values, values + count);
    double total = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        auto& value = values[i];
        value = std::exp(value - maximum);
        total += value;
    }
    for (std::size_t i = 0; i < count; ++i)
        values[i] /= total;
}

} // namespace

void applyActivation(std::vector<double>& values,
                     const std::string& activation)
{
    applyActivation(values.data(), values.size(), activation);
}

void applyActivation(double* values, std::size_t count, const std::string& activation)
{
    if (activation == "softmax") {
        softmax(values, count);
        return;
    }
    if (activation == "linear")
        return;
    if (activation != "relu" && activation != "sigmoid" && activation != "tanh")
        throw std::invalid_argument("Unknown activation: " + activation);

    for (std::size_t i = 0; i < count; ++i) {
        auto& value = values[i];
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
    std::vector<double> derivatives(activated.size(), 1.0);
    applyActivationDerivative(derivatives, activated, activation);
    return derivatives;
}

void applyActivationDerivative(std::vector<double>& deltas,
    const std::vector<double>& activated, const std::string& activation)
{
    if (deltas.size() != activated.size())
        throw std::invalid_argument("activation derivative dimensions do not match");
    applyActivationDerivative(deltas.data(), activated.data(), activated.size(), activation);
}

void applyActivationDerivative(double* deltas, const double* activated,
    std::size_t count, const std::string& activation)
{
    const auto apply = [&](auto derivative) {
        for (std::size_t i = 0; i < count; ++i)
            deltas[i] *= derivative(activated[i]);
    };

    if (activation == "sigmoid")
        apply([](double a) { return a * (1.0 - a); });
    else if (activation == "tanh")
        apply([](double a) { return 1.0 - a * a; });
    else if (activation == "relu")
        apply([](double a) { return a > 0.0 ? 1.0 : 0.0; });
    else if (activation == "linear")
        return;
    else if (activation == "softmax")
        throw std::invalid_argument("softmax has no element-wise derivative; "
            "its gradient is computed with the cross-entropy loss");
    else
        throw std::invalid_argument("Unknown activation: " + activation);
}
