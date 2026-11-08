#include "neural/ModelSerializer.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

namespace {
void close(double actual, double expected)
{
    test::check(std::isfinite(actual) && std::abs(actual - expected) < 1e-12,
        "Adam differs from independent scalar reference");
}

void same(const NeuralNetwork& a, const NeuralNetwork& b)
{
    test::check(a.layers().size() == b.layers().size(), "incorrect layer count");
    for (std::size_t i = 0; i < a.layers().size(); ++i) {
        test::check(a.layers()[i].weights == b.layers()[i].weights, "weights changed");
        test::check(a.layers()[i].biases == b.layers()[i].biases, "biases changed");
    }
}
}

int main()
{
    NetworkConfig config;
    config.layers = {2, 1};
    config.hidden_activation = "sigmoid";
    config.output_activation = "sigmoid";
    config.initialization = "xavier";
    config.optimizer = "adam";
    config.learning_rate = 0.001;
    Layer layer(2, 1);
    layer.weights = {{0.2, -0.3}};
    layer.biases = {0.1};
    NeuralNetwork network(config, {layer});
    auto indexed = network;
    // Different batches force moment accumulation, including a gradient sign change.
    const std::vector<TrainingSample> samples = {{{1.0, -2.0}, {1.0}}, {{-0.5, 0.3}, {0.0}}};
    double parameters[] = {0.2, -0.3, 0.1};
    double m[] = {0, 0, 0}, v[] = {0, 0, 0};
    for (unsigned step = 1; step <= 4; ++step) {
        const auto index = (step - 1) % 2;
        const auto& sample = samples[index];
        const double p = 1.0 / (1.0 + std::exp(-(parameters[0] * sample.inputs[0]
            + parameters[1] * sample.inputs[1] + parameters[2])));
        const double g[] = {(p - sample.target[0]) * sample.inputs[0],
            (p - sample.target[0]) * sample.inputs[1], p - sample.target[0]};
        const double loss = network.trainBatch({sample});
        close(loss, -sample.target[0] * std::log(p) - (1.0 - sample.target[0]) * std::log1p(-p));
        indexed.trainBatch(samples, {index}, 0, 1);
        same(network, indexed);
        for (unsigned k = 0; k < 3; ++k) {
            m[k] = 0.9 * m[k] + 0.1 * g[k];
            v[k] = 0.999 * v[k] + 0.001 * g[k] * g[k];
            parameters[k] -= 0.001 * (m[k] / (1 - std::pow(0.9, step)))
                / (std::sqrt(v[k] / (1 - std::pow(0.999, step))) + 1e-8);
        }
        close(network.layers()[0].weights[0][0], parameters[0]);
        close(network.layers()[0].weights[0][1], parameters[1]);
        close(network.layers()[0].biases[0], parameters[2]);
    }
    test::TempDirectory directory;
    const auto path = directory.path / "adam.nn";
    ModelSerializer::save(network, path.string());
    auto restored = ModelSerializer::load(path.string());
    test::check(restored.config().optimizer == "adam", "optimizer not restored");
    network.trainBatch(samples);
    restored.trainBatch(samples);
    same(network, restored); // Includes the persisted step and both moments.
    const auto before = network;
    test::expectThrow<std::invalid_argument>([&] { network.trainBatch({}); });
    same(network, before);
    network.trainBatch(samples);
    auto control = before;
    control.trainBatch(samples);
    same(network, control);

    // A zero gradient must stay zero, without 0/0.
    layer.weights = {{0, 0}};
    layer.biases = {0};
    NeuralNetwork zero(config, {layer});
    zero.trainBatch({{{0, 0}, {0.5}}});
    close(zero.layers()[0].biases[0], 0);

    // Old models have no optimizer key and load as SGD.
    ModelSerializer::save(zero, path.string());
    nlohmann::json model;
    { std::ifstream file(path); file >> model; }
    model.erase("optimizer_state");
    model["config"].erase("optimizer");
    { std::ofstream file(path); file << model; }
    test::check(ModelSerializer::load(path.string()).config().optimizer == "sgd", "legacy model rejected");
    model["config"]["optimizer"] = "invalid";
    { std::ofstream file(path); file << model; }
    test::expectThrow<std::runtime_error>([&] { ModelSerializer::load(path.string()); });
    std::cout << "Adam tests passed\n";
}
