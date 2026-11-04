#include "neural/ModelSerializer.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

namespace {

nlohmann::json snapshot(const NetworkConfig &config, const std::filesystem::path &path)
{
    ModelSerializer::save(NeuralNetwork(config), path.string());
    std::ifstream file(path);
    return nlohmann::json::parse(file);
}

NetworkConfig explicitConfig()
{
    NetworkConfig config;
    config.layers = {2, 2};
    config.hidden_activation = "relu";
    config.output_activation = "sigmoid";
    config.learning_rate = 0.5;
    config.initialization = "xavier";
    config.seed = 7;
    return config;
}

std::vector<Layer> explicitLayers()
{
    Layer layer(2, 2);
    layer.weights = {{1.0, -1.0}, {0.5, 2.0}};
    layer.biases = {0.0, -1.0};
    return {layer};
}

void testExplicitParameters()
{
    const NeuralNetwork network(explicitConfig(), explicitLayers());

    test::check(network.config().layers == explicitConfig().layers
        && network.config().learning_rate == 0.5 && network.config().seed == 7,
        "config() does not expose the network configuration");
    test::check(network.layers().size() == 1
        && network.layers()[0].weights == explicitLayers()[0].weights
        && network.layers()[0].biases == explicitLayers()[0].biases,
        "layers() does not expose the given parameters");

    // z = [2×1 + 1×(-1) + 0, 2×0.5 + 1×2 - 1] = [1, 2], then sigmoid.
    const auto outputs = network.predict({2.0, 1.0});
    const auto sigmoid = [](double z) { return 1.0 / (1.0 + std::exp(-z)); };
    test::check(outputs.size() == 2 && std::abs(outputs[0] - sigmoid(1.0)) < 1e-12
        && std::abs(outputs[1] - sigmoid(2.0)) < 1e-12,
        "prediction does not use the given parameters");
}

void testInvalidParameters()
{
    const auto rejects = [](NetworkConfig config, std::vector<Layer> layers) {
        test::expectThrow<std::invalid_argument>(
            [&] { NeuralNetwork network(std::move(config), std::move(layers)); });
    };
    auto config = explicitConfig();

    rejects(config, {});
    rejects(config, {explicitLayers()[0], explicitLayers()[0]});
    auto layers = explicitLayers();
    layers[0].weights.pop_back();
    rejects(config, layers);
    layers = explicitLayers();
    layers[0].biases.push_back(0.0);
    rejects(config, layers);
    layers = explicitLayers();
    layers[0].weights[1].push_back(0.0);
    rejects(config, layers);
    layers = explicitLayers();
    layers[0].weights[0][1] = std::nan("");
    rejects(config, layers);
    layers = explicitLayers();
    layers[0].biases[0] = INFINITY;
    rejects(config, layers);
    config.layers = {2};
    rejects(config, {});
    config.layers = {2, 0};
    rejects(config, {Layer(2, 0)});
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
    NeuralNetwork pendingTraining(config);
    const std::vector<TrainingSample> samples = {{{0.5}, {1.0}}};
    const auto prediction = pendingTraining.predict(samples[0].inputs);
    const auto updateError = test::expectThrow<std::runtime_error>([&] {
        pendingTraining.trainBatch(samples);
    });
    const auto lossError = test::expectThrow<std::runtime_error>([&] {
        pendingTraining.loss(samples);
    });
    test::check(updateError.find("not implemented") != std::string::npos
        && lossError.find("not implemented") != std::string::npos,
        "pending training methods must fail explicitly");
    test::check(pendingTraining.predict(samples[0].inputs) == prediction,
        "pending training methods changed predictions");
    config.initialization = "unknown";
    test::expectThrow<std::invalid_argument>([&] { NeuralNetwork network(config); });
    testExplicitParameters();
    testInvalidParameters();
    std::cout << "NeuralNetwork tests passed\n";
}
