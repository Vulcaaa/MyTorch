#include "Prediction.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <string_view>

int runPrediction(const NeuralNetwork& network, const Dataset& dataset)
{
    constexpr std::array<std::string_view, 3> classes = {
        "Nothing", "Check", "Checkmate"
    };
    for (const auto& sample : dataset) {
        const auto scores = network.predict(sample.inputs);
        const auto best = std::max_element(scores.begin(), scores.end());
        std::cout << classes[best - scores.begin()] << '\n';
    }
    return 0;
}
