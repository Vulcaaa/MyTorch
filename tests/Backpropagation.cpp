#include "neural/NeuralNetwork.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
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

Layer layerWith(std::vector<std::vector<double>> weights, std::vector<double> biases)
{
    Layer layer(weights.front().size(), weights.size());
    layer.weights = std::move(weights);
    layer.biases = std::move(biases);
    return layer;
}

NeuralNetwork networkWith(std::vector<std::size_t> sizes, const std::string &hidden,
    const std::string &output, std::vector<Layer> layers)
{
    NetworkConfig config;
    config.layers = std::move(sizes);
    config.hidden_activation = hidden;
    config.output_activation = output;
    return NeuralNetwork(config, std::move(layers));
}

void checkLayer(const Layer &actual, const Layer &expected, const std::string &message)
{
    test::check(actual.weights.size() == expected.weights.size()
        && actual.biases.size() == expected.biases.size(), message + ": wrong shape");
    for (std::size_t j = 0; j < expected.weights.size(); ++j) {
        test::check(actual.weights[j].size() == expected.weights[j].size(), message + ": wrong shape");
        for (std::size_t k = 0; k < expected.weights[j].size(); ++k)
            checkClose(actual.weights[j][k], expected.weights[j][k], message + ": weight");
        checkClose(actual.biases[j], expected.biases[j], message + ": bias");
    }
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

void testGradients()
{
    // No hidden layer: logits = [1 * 2 - 0.5 * 4, 0] = [0, 0], so p = [1/2, 1/2].
    // delta = p - y = [-1/2, 1/2]; weight gradient = delta[j] * input[k]; bias gradient = delta.
    const auto single = networkWith({2, 2}, "relu", "softmax", {layerWith({{1, -0.5}, {0, 0}}, {0, 0})});
    const TrainingSample first{{2, 4}, {1, 0}};
    const auto gradients = single.gradients({first});
    test::check(gradients.size() == 1, "one gradient layer per parameter layer");
    checkLayer(gradients[0], layerWith({{-1, -2}, {1, 2}}, {-0.5, 0.5}), "output layer gradient");

    // One sigmoid hidden neuron: input 2, z1 = (log 3 / 2) * 2 = log 3, h = sigmoid(z1) = 3/4,
    // f'(h) = 3/4 * 1/4 = 3/16. Output z2 = 4 * 3/4 - 3 = 0, p = 1/2, target 1: delta2 = -1/2.
    // Output: weight -1/2 * 3/4 = -3/8, bias -1/2.
    // Hidden: delta1 = 4 * -1/2 * 3/16 = -3/8; weight -3/8 * 2 = -3/4, bias -3/8.
    const auto hidden = networkWith({1, 1, 1}, "sigmoid", "sigmoid",
        {layerWith({{std::log(3.0) / 2}}, {0}), layerWith({{4}}, {-3})});
    const auto chained = hidden.gradients({{{2}, {1}}});
    checkLayer(chained[1], layerWith({{-0.375}}, {-0.5}), "output layer gradient");
    checkLayer(chained[0], layerWith({{-0.75}}, {-0.375}), "hidden layer gradient");

    // A batch gradient is the mean of the sample gradients.
    const TrainingSample second{{1, 0}, {0, 1}};
    const auto alone = single.gradients({second});
    const auto batch = single.gradients({first, second});
    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t k = 0; k < 2; ++k)
            checkClose(batch[0].weights[j][k],
                (gradients[0].weights[j][k] + alone[0].weights[j][k]) / 2, "batch weight mean");
        checkClose(batch[0].biases[j], (gradients[0].biases[j] + alone[0].biases[j]) / 2,
            "batch bias mean");
    }

    test::expectThrow<std::invalid_argument>([&] { single.gradients({}); });
    // Softmax has no element-wise derivative, so it cannot be a trainable hidden activation.
    const auto softmaxHidden = networkWith({1, 1, 1}, "softmax", "sigmoid",
        {layerWith({{1}}, {0}), layerWith({{1}}, {0})});
    test::expectThrow<std::invalid_argument>([&] { softmaxHidden.gradients({{{1}, {1}}}); });
    const auto overflowing = identityNetwork("softmax", 1e308);
    test::expectThrow<std::runtime_error>([&] { overflowing.gradients({{{10, 0}, {1, 0}}}); });
}

} // namespace

int main()
{
    testValidSamples();
    testInvalidSamples();
    testLoss();
    testGradients();
    std::cout << "Backpropagation tests passed\n";
}
