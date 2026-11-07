#pragma once

#include <string>
#include <vector>

void applyActivation(std::vector<double>& values, const std::string& activation);

// Derivative f'(z) of an element-wise activation, computed from its outputs a = f(z).
// ReLU uses 0 at z = 0. Softmax is rejected: its gradient is combined with the loss instead.
// Throws std::invalid_argument for softmax or an unknown activation.
std::vector<double> activationDerivative(const std::vector<double>& activated,
                                         const std::string& activation);

// Multiply existing deltas by f'(z) in place, without allocating a derivative vector.
void applyActivationDerivative(std::vector<double>& deltas,
    const std::vector<double>& activated, const std::string& activation);
