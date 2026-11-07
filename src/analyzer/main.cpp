#include <chrono>
#include <iostream>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

#include "Arguments.hpp"
#include "Prediction.hpp"
#include "Training.hpp"
#include "logging/Logger.hpp"
#include "neural/ModelSerializer.hpp"

namespace {

void validateChessModel(const NeuralNetwork& network)
{
    if (network.inputSize() != 777 || network.outputSize() != 3)
        throw std::invalid_argument("chess model requires 777 inputs and 3 outputs; got "
            + std::to_string(network.inputSize()) + " inputs and "
            + std::to_string(network.outputSize()) + " outputs");
}

// Records the run in logs/train_<UTC time>.log, including a failure, which is rethrown.
int trainModel(const AnalyzerArguments& arguments)
{
    const auto start = std::chrono::steady_clock::now();
    const auto elapsed = [&start] { return formatSeconds(std::chrono::steady_clock::now() - start); };
    Logger log("logs", "train", std::cerr);
    log.info("Training started");
    try {
        NeuralNetwork network = ModelSerializer::load(arguments.loadFile);
        validateChessModel(network);
        log.info("Model " + arguments.loadFile + ": " + describeNetworkConfig(network.config()));
        const TrainingResult result = runTraining(network, arguments, std::cerr, log);
        log.info("Training complete in " + elapsed() + ": epochs_completed="
            + std::to_string(result.epochsCompleted) + " best_epoch=" + std::to_string(result.bestEpoch));
    } catch (const std::exception& error) {
        log.error("Training failed after " + elapsed() + ": " + error.what());
        throw;
    }
    return 0;
}

int runAnalyzer(const AnalyzerArguments& arguments)
{
    if (arguments.mode == AnalyzerMode::Train)
        return trainModel(arguments);
    NeuralNetwork network = ModelSerializer::load(arguments.loadFile);
    validateChessModel(network);
    return runPrediction(network, loadDataset(arguments.chessFile, DatasetMode::Prediction));
}

} // namespace

int main(int ac, char** av)
{
    if (ac == 2 && std::string_view(av[1]) == "--help") {
        std::cout << "USAGE\n"
                  << "\t./my_torch_analyzer --predict LOADFILE CHESSFILE\n"
                  << "\t./my_torch_analyzer --train [OPTIONS] LOADFILE CHESSFILE\n\n"
                  << "Train a neural network or classify chess positions in FEN format.\n\n"
                  << "TRAINING OPTIONS (before LOADFILE and CHESSFILE, in any order)\n"
                  << "\t--validation FILE  Independent validation dataset (default: seeded 90/10 split)\n"
                  << "\t--epochs N         Maximum epochs (positive integer; default: 100)\n"
                  << "\t--batch-size N     Mini-batch size (positive integer; default: 32)\n"
                  << "\t--threads N        BLAS thread limit (positive integer; default: library/environment)\n"
                  << "\t--patience N       Early-stopping patience (positive integer; default: 10)\n"
                  << "\t--min-delta X      Minimum loss improvement (finite, >= 0; default: 0.0001)\n"
                  << "\t--history FILE     Override automatic per-run history CSV path\n"
                  << "\t--save FILE        Save trained model here (default: LOADFILE)\n\n"
                  << "Learning rate and seed come from the loaded model.\n"
                  << "Each training run is logged in logs/train_<UTC time>.log.\n";
        return 0;
    }

    try {
        return runAnalyzer(parseAnalyzerArguments(ac, av));
    } catch (const std::exception& error) {
        std::cerr << "MyTorch: " << error.what() << '\n';
        return 84;
    }
}
