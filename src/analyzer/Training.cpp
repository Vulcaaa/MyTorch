#include "Training.hpp"
#include "neural/ModelSerializer.hpp"

#include <stdexcept>

void saveTrainingResult(const NeuralNetwork& network, const AnalyzerArguments& arguments)
{
    if (arguments.mode != AnalyzerMode::Train) {
        throw std::invalid_argument("saving is only valid in training mode");
    }
    const std::string& destination = arguments.saveFile.empty()
        ? arguments.loadFile : arguments.saveFile;
    ModelSerializer::save(network, destination);
}
