#include "neural/NeuralNetwork.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

NeuralNetwork makeNetwork(const std::string &outputActivation)
{
    NetworkConfig config;
    config.layers = {3, 2};
    config.hidden_activation = "relu";
    config.output_activation = outputActivation;
    config.initialization = "xavier";
    return NeuralNetwork(config);
}

// No hidden layer and identity weights: the output logits equal the inputs.
NeuralNetwork identityNetwork(const std::string &outputActivation, double firstWeight = 1)
{
    NetworkConfig config;
    config.layers = {2, 2};
    config.output_activation = outputActivation;
    Layer layer(2, 2);
    layer.weights = {{firstWeight, 0}, {0, 1}};
    return NeuralNetwork(config, {layer});
}

void checkClose(double actual, double expected, const std::string &message)
{
    test::check(std::isfinite(actual) && std::abs(actual - expected) < 1e-12, message);
}

void rejects(const NeuralNetwork &network, const std::vector<TrainingSample> &samples)
{
    test::expectThrow<std::invalid_argument>([&] { network.validateSamples(samples); });
}

void testValidSamples()
{
    makeNetwork("softmax").validateSamples({
        {{1, 0, 1}, {1, 0}},
        {{0.5, -2, 3}, {0.2, 0.8}},
        {{0, 0, 0}, {0.5, 0.5 + 1e-9}}
    });
    makeNetwork("sigmoid").validateSamples({
        {{1, 0, 1}, {1, 1}},
        {{1, 0, 1}, {0, 0}},
        {{1, 0, 1}, {0.3, 0.9}}
    });
}

void testInvalidSamples()
{
    const auto softmax = makeNetwork("softmax");
    const auto sigmoid = makeNetwork("sigmoid");

    rejects(softmax, {});
    rejects(softmax, {{{1, 0}, {1, 0}}});
    rejects(softmax, {{{1, 0, 1}, {1, 0, 0}}});
    rejects(softmax, {{{1, std::nan(""), 1}, {1, 0}}});
    rejects(softmax, {{{1, INFINITY, 1}, {1, 0}}});
    rejects(softmax, {{{1, 0, 1}, {std::nan(""), 1}}});
    rejects(softmax, {{{1, 0, 1}, {-0.5, 1.5}}});
    rejects(softmax, {{{1, 0, 1}, {0.5, 0.4}}});
    rejects(sigmoid, {{{1, 0, 1}, {1.5, 0}}});
    rejects(sigmoid, {{{1, 0, 1}, {0, -0.1}}});

    const auto message = test::expectThrow<std::invalid_argument>([&] {
        softmax.validateSamples({{{1, 0, 1}, {1, 0}}, {{1, 0}, {1, 0}}});
    });
    test::check(message.find("training sample 1") != std::string::npos,
        "error does not identify the invalid sample");
    rejects(makeNetwork("linear"), {{{1, 0, 1}, {1, 0}}});
}

void testLoss()
{
    // softmax([0, 0]) = [1/2, 1/2] and softmax([0, log 3]) = [1/4, 3/4]: the loss is the mean.
    const auto softmax = identityNetwork("softmax");
    checkClose(softmax.loss({{{0, 0}, {1, 0}}, {{0, std::log(3.0)}, {0, 1}}}),
        (std::log(2.0) - std::log(0.75)) / 2, "loss must average the sample losses");
    // sigmoid(0) = 1/2 on both outputs: log 2 + log 2.
    checkClose(identityNetwork("sigmoid").loss({{{0, 0}, {1, 0}}}), 2 * std::log(2.0),
        "sigmoid loss must sum over outputs");

    // With hidden layers, the loss matches the predictions: mean of -log(expected probability).
    NetworkConfig config;
    config.layers = {3, 4, 3};
    config.hidden_activation = "tanh";
    config.output_activation = "softmax";
    config.initialization = "xavier";
    config.seed = 3;
    const NeuralNetwork network(config);
    const std::vector<TrainingSample> samples{
        {{1, 0, -1}, {1, 0, 0}}, {{0.5, 2, 0}, {0, 0, 1}}, {{-1, -1, 3}, {0, 1, 0}}};
    double total = 0.0;
    for (const auto &sample : samples) {
        const auto probabilities = network.predict(sample.inputs);
        for (std::size_t k = 0; k < probabilities.size(); ++k)
            total -= sample.target[k] * std::log(probabilities[k]);
    }
    checkClose(network.loss(samples), total / samples.size(), "loss must match predictions");

    // Samples are validated first.
    test::expectThrow<std::invalid_argument>([&] { softmax.loss({}); });
    test::expectThrow<std::invalid_argument>([&] { softmax.loss({{{0}, {1, 0}}}); });
    // A logit of 10 * 1e308 overflows to infinity, so the loss is not finite.
    const auto overflowing = identityNetwork("softmax", 1e308);
    test::expectThrow<std::runtime_error>([&] { overflowing.loss({{{10, 0}, {1, 0}}}); });
}

} // namespace

int main()
{
    testValidSamples();
    testInvalidSamples();
    testLoss();
    std::cout << "Backpropagation tests passed\n";
}
