#include "neural/ModelSerializer.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>

namespace {

void checkScores(const std::vector<double>& actual, const std::vector<double>& expected)
{
    test::check(actual.size() == expected.size(), "incorrect prediction size");
    for (std::size_t i = 0; i < actual.size(); ++i)
        test::check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 1e-12,
            "prediction differs from hand calculation");
}

} // namespace

int main()
{
    test::TempDirectory directory;
    const auto source = directory.path / "known.nn";
    const auto restored = directory.path / "restored.nn";
    NetworkConfig config;
    config.layers = {2, 2, 3};
    config.hidden_activation = "relu";
    config.output_activation = "softmax";
    config.initialization = "xavier";
    ModelSerializer::save(NeuralNetwork(config), source.string());
    nlohmann::json model;
    {
        std::ifstream file(source);
        file >> model;
    }
    // Input [2, -1] gives hidden preactivations [-1, 3], then ReLU [0, 3].
    model["layers"][0]["weights"] = {{1.0, 2.0}, {-1.0, -3.0}};
    model["layers"][0]["biases"] = {-1.0, 2.0};
    // Output logits [0, log(2), log(3)] give probabilities [1/6, 2/6, 3/6].
    model["layers"][1]["weights"] = {{2.0, 1.0}, {-1.0, 0.0}, {4.0, -1.0}};
    model["layers"][1]["biases"] = {-3.0, std::log(2.0), 3.0 + std::log(3.0)};
    {
        std::ofstream file(source);
        file << model.dump(2);
    }
    const auto network = ModelSerializer::load(source.string());
    const std::vector<double> inputs{2, -1};
    const auto scores = network.predict(inputs);
    checkScores(scores, {1.0 / 6, 2.0 / 6, 3.0 / 6});
    checkScores(network.predict(inputs), scores);
    ModelSerializer::save(network, restored.string());
    const auto reloaded = ModelSerializer::load(restored.string());
    test::check(reloaded.predict(inputs) == scores, "save/load changed predictions");
    for (const auto& invalid : std::vector<std::vector<double>>{{}, {1}, {1, 2, 3}})
        test::expectThrow<std::runtime_error>([&] { network.predict(invalid); });
    checkScores(network.predict(inputs), scores);

    // A network without hidden layers must apply its output activation directly.
    model["config"]["layers"] = {2, 1};
    model["config"]["output_activation"] = "sigmoid";
    model["layers"] = nlohmann::json::array({{
        {"weights", {{2.0, -1.0}}}, {"biases", {-5.0}}
    }});
    {
        std::ofstream file(source);
        file << model.dump(2);
    }
    checkScores(ModelSerializer::load(source.string()).predict(inputs), {0.5});
    std::cout << "Prediction tests passed\n";
}
