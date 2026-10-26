#pragma once

#include <optional>
#include <string>
#include <vector>

struct DatasetSample {
    std::vector<double> inputs; // 777 encoded FEN values.
    std::optional<std::vector<double>> target; // Nothing, Check, Checkmate; absent if unlabeled.
};

using Dataset = std::vector<DatasetSample>;

enum class DatasetMode {
    Prediction,
    Training
};

Dataset loadDataset(const std::string& path, DatasetMode mode = DatasetMode::Prediction);
