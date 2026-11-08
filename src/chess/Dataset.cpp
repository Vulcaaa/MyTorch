#include "Dataset.hpp"
#include "Encoding.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

DatasetSample parseSample(std::istringstream& stream, DatasetMode mode)
{
    std::string fen;
    for (int field = 0; field < 6; ++field) {
        std::string value;
        if (!(stream >> value)) {
            throw std::invalid_argument("expected six FEN fields");
        }
        if (field != 0) {
            fen += ' ';
        }
        fen += value;
    }

    DatasetSample sample{encodeFEN(fen), std::nullopt};
    std::string label;
    if (!(stream >> label)) {
        if (mode == DatasetMode::Training) {
            throw std::invalid_argument("missing training label");
        }
        return sample;
    }

    std::size_t index;
    if (label == "Nothing") {
        index = 0;
    } else if (label == "Check") {
        index = 1;
    } else if (label == "Checkmate") {
        index = 2;
    } else {
        throw std::invalid_argument("unknown label: " + label);
    }

    std::string suffix;
    if (stream >> suffix) {
        if (index == 0 || (suffix != "White" && suffix != "Black")) {
            throw std::invalid_argument("invalid label suffix: " + suffix);
        }
        std::string extra;
        if (stream >> extra) {
            throw std::invalid_argument("unexpected data after label: " + extra);
        }
    }
    sample.target = std::vector<double>(3, 0.0);
    (*sample.target)[index] = 1.0;
    return sample;
}

} // namespace

Dataset loadDataset(const std::string& path, DatasetMode mode)
{
    Dataset dataset;
    readDataset(path, mode, [&](DatasetSample sample) { dataset.push_back(std::move(sample)); });
    return dataset;
}

void readDataset(const std::string& path, DatasetMode mode,
    const std::function<void(DatasetSample)>& consume)
{
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error(path + ": cannot open dataset");
    }

    std::size_t samples = 0;
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        std::istringstream stream(line);
        stream >> std::ws;
        if (stream.eof()) {
            continue;
        }
        try {
            consume(parseSample(stream, mode));
            ++samples;
        } catch (const std::invalid_argument& error) {
            throw std::runtime_error(path + ":" + std::to_string(lineNumber)
                + ": " + error.what());
        }
    }
    if (file.bad() || (file.fail() && !file.eof())) {
        throw std::runtime_error(path + ":" + std::to_string(lineNumber + 1)
            + ": cannot read dataset");
    }
    if (samples == 0) {
        throw std::runtime_error(path + ": dataset contains no positions");
    }
}
