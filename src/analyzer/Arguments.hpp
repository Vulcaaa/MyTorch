#pragma once

#include <string>

enum class AnalyzerMode {
    Predict,
    Train
};

struct AnalyzerArguments {
    AnalyzerMode mode;
    std::string loadFile;
    std::string chessFile;
    std::string saveFile; // Empty in prediction mode; defaults to loadFile in training mode.
};

// Throws std::invalid_argument for invalid command-line arguments.
AnalyzerArguments parseAnalyzerArguments(int argc, const char* const argv[]);
