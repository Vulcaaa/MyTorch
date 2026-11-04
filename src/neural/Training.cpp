#include "NeuralNetwork.hpp"

#include <stdexcept>

// Issue #5 replaces these explicit failures with the neural training calculations.
double NeuralNetwork::trainBatch(const std::vector<TrainingSample>&)
{
    throw std::runtime_error("trainBatch is not implemented yet (issue #5)");
}

double NeuralNetwork::loss(const std::vector<TrainingSample>&) const
{
    throw std::runtime_error("loss is not implemented yet (issue #5)");
}
