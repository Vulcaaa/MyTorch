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
    const std::string &output, std::vector<Layer> layers, double learningRate = 0.01)
{
    NetworkConfig config;
    config.layers = std::move(sizes);
    config.hidden_activation = hidden;
    config.output_activation = output;
    config.learning_rate = learningRate;
    return NeuralNetwork(config, std::move(layers));
}

bool sameParameters(const std::vector<Layer> &left, const std::vector<Layer> &right)
{
    if (left.size() != right.size())
        return false;
    for (std::size_t i = 0; i < left.size(); ++i)
        if (left[i].weights != right[i].weights || left[i].biases != right[i].biases)
            return false;
    return true;
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

void testTrainBatch()
{
    // Same sample as testGradients: loss log 2, weight gradient [[-1, -2], [1, 2]],
    // bias gradient [-1/2, 1/2]. With learning rate 0.1: parameter - 0.1 * gradient.
    const std::vector<Layer> initial{layerWith({{1, -0.5}, {0, 0}}, {0, 0})};
    const TrainingSample sample{{2, 4}, {1, 0}};
    auto network = networkWith({2, 2}, "relu", "softmax", initial, 0.1);
    const auto checkpoint = network;
    checkClose(network.trainBatch({sample}), std::log(2.0), "trainBatch must return the loss before the update");
    checkLayer(network.layers()[0], layerWith({{1.1, -0.3}, {-0.1, -0.2}}, {0.05, -0.05}),
        "updated parameters");
    test::check(network.loss({sample}) < std::log(2.0), "an update must lower the batch loss");
    test::check(sameParameters(checkpoint.layers(), initial), "a copy must not follow the updates");
    test::check(network.config().learning_rate == 0.1, "training must not change the configuration");

    // Every failure leaves the network unchanged.
    const auto unchanged = network.layers();
    test::expectThrow<std::invalid_argument>([&] { network.trainBatch({}); });
    test::check(sameParameters(network.layers(), unchanged), "invalid batch changed the network");
    for (const double rate : {0.0, -0.1, std::nan("")}) {
        auto invalidRate = networkWith({2, 2}, "relu", "softmax", initial, rate);
        test::expectThrow<std::invalid_argument>([&] { invalidRate.trainBatch({sample}); });
        test::check(sameParameters(invalidRate.layers(), initial), "invalid learning rate changed the network");
    }
    // 1e308 * gradient 2 overflows to infinity.
    auto overflowing = networkWith({2, 2}, "relu", "softmax", initial, 1e308);
    test::expectThrow<std::runtime_error>([&] { overflowing.trainBatch({sample}); });
    test::check(sameParameters(overflowing.layers(), initial), "overflowing update changed the network");
}

// Measures each gradient directly: nudge one parameter by +h and -h, and divide the loss
// change by 2h. Backpropagation must agree for every weight and bias and every activation
// pair. Inputs and the seed keep ReLU pre-activations away from its kink at 0.
void testGradientsMatchFiniteDifferences()
{
    constexpr double h = 1e-6;
    const std::vector<TrainingSample> samples{
        {{0.3, -1.2, 0.8}, {0, 1, 0}}, {{1.5, 0.2, -0.4}, {0.2, 0.3, 0.5}}};

    for (const auto *hidden : {"sigmoid", "tanh", "relu", "linear"}) {
        for (const auto *output : {"softmax", "sigmoid"}) {
            NetworkConfig config;
            config.layers = {3, 4, 5, 3};
            config.hidden_activation = hidden;
            config.output_activation = output;
            config.initialization = "xavier";
            config.seed = 11;
            const NeuralNetwork network(config);
            const auto gradients = network.gradients(samples);
            auto layers = network.layers();

            const auto measure = [&](double &parameter) {
                const double saved = parameter;
                parameter = saved + h;
                const double up = NeuralNetwork(config, layers).loss(samples);
                parameter = saved - h;
                const double down = NeuralNetwork(config, layers).loss(samples);
                parameter = saved;
                return (up - down) / (2 * h);
            };
            const auto where = std::string(hidden) + "/" + output;
            for (std::size_t i = 0; i < layers.size(); ++i) {
                for (std::size_t j = 0; j < layers[i].weights.size(); ++j) {
                    for (std::size_t k = 0; k < layers[i].weights[j].size(); ++k)
                        test::check(std::abs(measure(layers[i].weights[j][k])
                            - gradients[i].weights[j][k]) < 1e-7, where + ": weight gradient mismatch");
                    test::check(std::abs(measure(layers[i].biases[j]) - gradients[i].biases[j]) < 1e-7,
                        where + ": bias gradient mismatch");
                }
            }
        }
    }
}

// XOR (output 1 when exactly one input is 1) needs a hidden layer: no single weighted sum
// separates its classes. Mini-batches of two samples must drive the loss close to 0 and
// classify all four cases, while a network without hidden layer stays near chance.
void testLearnsXor()
{
    for (const std::string output : {"softmax", "sigmoid"}) {
        const bool softmax = output == "softmax";
        std::vector<TrainingSample> samples;
        for (const double a : {0.0, 1.0})
            for (const double b : {0.0, 1.0}) {
                const double y = a != b ? 1.0 : 0.0;
                samples.push_back({{a, b}, softmax ? std::vector<double>{1 - y, y}
                                                   : std::vector<double>{y}});
            }
        const std::vector<TrainingSample> first(samples.begin(), samples.begin() + 2);
        const std::vector<TrainingSample> second(samples.begin() + 2, samples.end());
        const std::size_t outputs = softmax ? 2 : 1;

        const auto train = [&](std::vector<std::size_t> layers) {
            NetworkConfig config;
            config.layers = std::move(layers);
            config.hidden_activation = "tanh";
            config.output_activation = output;
            config.learning_rate = 0.5;
            config.initialization = "xavier";
            NeuralNetwork network(config);
            for (int step = 0; step < 1000; ++step)
                network.trainBatch(step % 2 == 0 ? first : second);
            return network;
        };

        const auto network = train({2, 4, outputs});
        test::check(network.loss(samples) < 0.05, output + ": XOR loss must drop close to 0");
        for (const auto &sample : samples) {
            const auto prediction = network.predict(sample.inputs);
            const bool predictedOne = softmax ? prediction[1] > prediction[0] : prediction[0] > 0.5;
            const bool expectedOne = sample.target.back() == 1.0;
            test::check(predictedOne == expectedOne, output + ": XOR case misclassified");
        }
        test::check(train({2, outputs}).loss(samples) > 0.6,
            output + ": XOR must not be learnable without a hidden layer");
    }
}

} // namespace

int main()
{
    testValidSamples();
    testInvalidSamples();
    testLoss();
    testGradients();
    testTrainBatch();
    testGradientsMatchFiniteDifferences();
    testLearnsXor();
    std::cout << "Backpropagation tests passed\n";
}
