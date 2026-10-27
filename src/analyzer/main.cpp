#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string_view>

#include "Arguments.hpp"
#include "NeuralNetwork.hpp"
#include "Dataset.hpp"
#include "ModelSerializer.hpp"

int runPrediction(const NeuralNetwork& nn, const Dataset& dataset)
{
    constexpr std::array<std::string_view, 3> classes = {
        "Nothing", "Check", "Checkmate"
    };

    for (const auto& sample : dataset) {
        const auto scores = nn.predict(sample.inputs);
        if (scores.size() != classes.size())
            throw std::invalid_argument("prediction requires exactly three output scores");
        const auto best = std::max_element(scores.begin(), scores.end());
        std::cout << classes[best - scores.begin()] << '\n';
    }
    return 0;
}

int runAnalyzer(AnalyzerArguments arguments)
{
    NeuralNetwork nn = ModelSerializer::load(arguments.loadFile);

    const auto mode = arguments.mode == AnalyzerMode::Train
        ? DatasetMode::Training : DatasetMode::Prediction;
    const Dataset dataset = loadDataset(arguments.chessFile, mode);

    if (arguments.mode == AnalyzerMode::Predict) {
        return runPrediction(nn, dataset);
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
                  << "Training is not implemented yet.\n";
        return 0;
    }

    try {
        return runAnalyzer(parseAnalyzerArguments(ac, av));
    } catch (const std::exception& error) {
        std::cerr << "MyTorch: " << error.what() << '\n';
        return 84;
    }
}
