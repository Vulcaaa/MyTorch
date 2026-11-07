#pragma once

#include <cstddef>
#include <string>

enum class AnalyzerMode {
    Predict,
    Train
};

struct TrainingArguments {
    std::string validationFile; // Empty selects a seeded, stratified 90/10 split of CHESSFILE.
    std::string historyFile; // Empty selects an automatic per-run path during training.
    std::size_t epochs = 100;
    std::size_t batchSize = 32;
    std::size_t patience = 10;
    double minDelta = 0.0001;
    std::size_t threads = 0; // Zero means preserve BLAS library/environment defaults.
};

struct AnalyzerArguments {
    AnalyzerMode mode;
    std::string loadFile;
    std::string chessFile;
    std::string saveFile; // Empty in prediction mode; defaults to loadFile in training mode.
    TrainingArguments training{};
};

// Throws std::invalid_argument for invalid command-line arguments.
AnalyzerArguments parseAnalyzerArguments(int argc, const char* const argv[]);
