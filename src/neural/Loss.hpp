#pragma once

#include <string>
#include <vector>

// Loss of one sample, computed from the output logits z (output values before the output
// activation) so it stays finite when a probability would round to 0 or 1. Never negative.
//   softmax: categorical cross-entropy, -sum_k y_k log(softmax(z)_k)
//   sigmoid: binary cross-entropy summed over outputs,
//            -sum_k [y_k log(sigmoid(z_k)) + (1 - y_k) log(1 - sigmoid(z_k))]
// Throws std::invalid_argument when logits is empty, the sizes differ, or the output
// activation has no loss (only softmax and sigmoid do).
double sampleLoss(const std::vector<double>& logits, const std::vector<double>& target,
                  const std::string& outputActivation);
double sampleLoss(const double* logits, const double* target, std::size_t count,
                  const std::string& outputActivation);
