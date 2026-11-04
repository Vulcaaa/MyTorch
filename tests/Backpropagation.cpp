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

} // namespace

int main()
{
    testValidSamples();
    testInvalidSamples();
    std::cout << "Backpropagation tests passed\n";
}
