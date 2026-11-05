#pragma once

#include "Arguments.hpp"
#include "neural/NeuralNetwork.hpp"

#include <ostream>

struct TrainingResult {
    std::size_t epochsCompleted = 0;
    std::size_t bestEpoch = 0;
    double bestValidationLoss = 0;
    bool earlyStopped = false;
    std::string historyFile;
};

// Loads labeled data (explicit validation or an automatic 90/10 split), trains a
// working copy, restores its best checkpoint,
// exports history and saves. On failure, the caller's network is unchanged.
TrainingResult runTraining(NeuralNetwork& network, const AnalyzerArguments& arguments,
    std::ostream& progress);

// Call only after training has completed successfully.
void saveTrainingResult(const NeuralNetwork& network, const AnalyzerArguments& arguments);
