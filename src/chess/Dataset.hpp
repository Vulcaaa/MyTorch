#pragma once

#include <optional>
#include <functional>
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

// Reads one sample at a time, avoiding a second full dense dataset during training.
void readDataset(const std::string& path, DatasetMode mode,
    const std::function<void(DatasetSample)>& consume);
