#include "mytorch/ModelSerializer.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

namespace {

nlohmann::json snapshot(const NetworkConfig &config, const std::filesystem::path &path)
{
    ModelSerializer::save(NeuralNetwork(config), path.string());
    std::ifstream file(path);
    return nlohmann::json::parse(file);
}

} // namespace

int main()
{
    test::TempDirectory directory;
    const auto path = directory.path / "network.nn";
    NetworkConfig config;
    config.layers = {4, 3, 2};
    config.hidden_activation = "relu";
    config.output_activation = "sigmoid";
    config.initialization = "he";
    for (const auto *method : {"he", "xavier"}) {
        config.initialization = method;
        config.seed = 42;
        const auto model = snapshot(config, path);
        test::check(snapshot(config, path) == model, "network construction is not reproducible");
        ++config.seed;
        test::check(snapshot(config, path).at("layers") != model.at("layers"),
            "different seeds produced the same parameters");
        const auto &layers = model.at("layers");
        test::check(layers.size() == config.layers.size() - 1, "incorrect parameter layer count");
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto &weights = layers[i].at("weights");
            const auto &biases = layers[i].at("biases");
            test::check(weights.size() == config.layers[i + 1], "incorrect neuron count");
            test::check(biases.size() == weights.size(), "incorrect bias count");
            for (const auto &row : weights) {
                test::check(row.size() == config.layers[i], "incorrect input count");
                for (const auto &weight : row)
                    test::check(std::isfinite(weight.get<double>()), "non-finite network weight");
            }
            for (const auto &bias : biases)
                test::check(bias.get<double>() == 0.0, "initial bias is not zero");
        }
    }
    config.layers = {1, 1};
    test::check(snapshot(config, path).at("layers").size() == 1, "minimal network rejected");
    for (const auto &sizes : std::vector<std::vector<std::size_t>>{
        {}, {1}, {0, 1}, {1, 0}, {1, 0, 1}}) {
        config.layers = sizes;
        test::expectThrow<std::invalid_argument>([&] { NeuralNetwork network(config); });
    }
    config.layers = {1, 1};
    config.initialization = "unknown";
    test::expectThrow<std::invalid_argument>([&] { NeuralNetwork network(config); });
    std::cout << "NeuralNetwork tests passed\n";
}
