#include "Prediction.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string_view>

int runPrediction(const NeuralNetwork& network, const Dataset& dataset)
{
    constexpr std::array<std::string_view, 3> classes = {
        "Nothing", "Check", "Checkmate"
    };
    std::size_t labeled = 0;
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        const auto scores = network.predict(sample.inputs);
        const auto best = std::max_element(scores.begin(), scores.end());
        std::cout << classes[best - scores.begin()] << '\n';
        if (sample.target) {
            ++labeled;
            const auto expected = std::max_element(sample.target->begin(), sample.target->end());
            correct += (best - scores.begin() == expected - sample.target->begin());
        }
    }
    std::ostringstream summary;
    summary << "Prediction summary: positions=" << dataset.size() << " labeled=" << labeled;
    if (labeled)
        summary << " correct=" << correct << '/' << labeled << " accuracy="
            << std::fixed << std::setprecision(2) << 100.0 * correct / labeled << '%';
    std::cerr << summary.str() << '\n';
    return 0;
}
