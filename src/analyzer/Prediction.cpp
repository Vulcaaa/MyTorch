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
        const auto index = best - scores.begin();
        std::cout << classes[index];
        if (index != 0) {
            // FEN input 768 is 1 for Black to move. The opponent gives check.
            std::cout << (sample.inputs[768] == 1.0 ? " White" : " Black");
        }
        std::cout << '\n';
    }
    return 0;
}
