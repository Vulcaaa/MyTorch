#include <iostream>
#include <string_view>

#include "Arguments.hpp"
#include "../neural/NetworkConfig.hpp"
#include "../neural/NeuralNetwork.hpp"
#include "../chess/Dataset.hpp"

int runAnalyzer(AnalyzerArguments arguments)
{
    NetworkConfig config = loadNetworkConfig(arguments.loadFile);
    NeuralNetwork neuralNetwork(config);

    const auto mode = arguments.mode == AnalyzerMode::Train
        ? DatasetMode::Training : DatasetMode::Prediction;
    const Dataset dataset = loadDataset(arguments.chessFile, mode);

    if (arguments.mode == AnalyzerMode::Predict) {
        // code
    } else if (arguments.mode == AnalyzerMode::Train) {
        // code
    }
    return 0;
}

int main(int ac, char** av)
{
    if (ac == 2 && std::string_view(av[1]) == "--help") {
        std::cout << "USAGE\n"
                  << "\t./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE\n\n"
                  << "Train a neural network or classify chess positions in FEN format.\n"
                  << "Training and prediction are not implemented yet.\n";
        return 0;
    }

    try {
        return runAnalyzer(parseAnalyzerArguments(ac, av));
    } catch (const std::exception& error) {
        std::cerr << "MyTorch: " << error.what() << '\n';
        return 84;
    }
}
