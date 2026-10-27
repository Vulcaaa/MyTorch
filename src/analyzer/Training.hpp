#pragma once

#include "Arguments.hpp"
#include "neural/NeuralNetwork.hpp"

// Call only after training has completed successfully.
void saveTrainingResult(const NeuralNetwork& network, const AnalyzerArguments& arguments);
