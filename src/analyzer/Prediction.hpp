#pragma once

#include "chess/Dataset.hpp"
#include "neural/NeuralNetwork.hpp"

// Prints one class per position, in dataset order.
int runPrediction(const NeuralNetwork& network, const Dataset& dataset);
