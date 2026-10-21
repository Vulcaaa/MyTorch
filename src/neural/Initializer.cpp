#include "Initializer.hpp"
#include "NeuralNetwork.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

template<typename Distribution>
void fillWeights(Layer &layer, Distribution &distribution, std::mt19937 &rng)
{
    for (auto &neuron : layer.weights)
        for (double &weight : neuron)
            weight = distribution(rng);
}

void initializeXavier(Layer &layer, std::mt19937 &rng)
{
    const double fanIn = static_cast<double>(layer.weights.front().size());
    const double fanOut = static_cast<double>(layer.weights.size());
    // Xavier uniform, gain = 1.
    const double bound = std::sqrt(6.0 / (fanIn + fanOut));
    std::uniform_real_distribution<double> distribution(-bound, bound);
    fillWeights(layer, distribution, rng);
}

void initializeHe(Layer &layer, std::mt19937 &rng)
{
    const double fanIn = static_cast<double>(layer.weights.front().size());
    // He normal, fan-in scaling for ReLU.
    std::normal_distribution<double> distribution(0.0, std::sqrt(2.0 / fanIn));
    fillWeights(layer, distribution, rng);
}

void validateLayer(const Layer &layer)
{
    if (layer.weights.empty() || layer.weights.front().empty()
        || layer.biases.size() != layer.weights.size())
        throw std::invalid_argument("initialization requires a non-empty layer with one bias per neuron");
    const auto inputs = layer.weights.front().size();
    for (const auto &neuron : layer.weights)
        if (neuron.size() != inputs)
            throw std::invalid_argument("initialization requires equal input counts for all neurons");
}

} // namespace

namespace initialization {

Method parseMethod(const std::string &name)
{
    if (name == "xavier")
        return Method::Xavier;
    if (name == "he")
        return Method::He;
    throw std::invalid_argument("unknown initialization method '" + name + "'");
}

void initialize(Layer &layer, Method method, std::mt19937 &rng)
{
    validateLayer(layer);
    switch (method) {
        case Method::Xavier:
            initializeXavier(layer, rng);
            break;
        case Method::He:
            initializeHe(layer, rng);
            break;
        default:
            throw std::invalid_argument("unknown initialization method");
    }
    std::fill(layer.biases.begin(), layer.biases.end(), 0.0);
}

} // namespace initialization
