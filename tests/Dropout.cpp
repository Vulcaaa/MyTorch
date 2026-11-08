#include "neural/ModelSerializer.hpp"
#include "neural/NeuralNetwork.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

NetworkConfig config(double dropout, const std::string& activation = "relu")
{
    NetworkConfig result;
    result.layers = {4, 6, 5, 3};
    result.hidden_activation = activation;
    result.output_activation = "softmax";
    result.initialization = "xavier";
    result.learning_rate = 0.05;
    result.seed = 11;
    result.dropout = dropout;
    return result;
}

std::vector<TrainingSample> samples()
{
    return {{{1, 0, 0, 1}, {1, 0, 0}}, {{0, 1, 1, 0}, {0, 1, 0}},
        {{1, 1, 0, 0}, {0, 0, 1}}, {{0, 0, 1, 1}, {1, 0, 0}}};
}

bool sameGradients(const std::vector<Layer>& left, const std::vector<Layer>& right)
{
    for (std::size_t i = 0; i < left.size(); ++i)
        if (left[i].weights != right[i].weights || left[i].biases != right[i].biases)
            return false;
    return true;
}

bool finite(const std::vector<Layer>& layers)
{
    for (const auto& layer : layers) {
        for (const auto& row : layer.weights)
            for (const auto value : row)
                if (!std::isfinite(value))
                    return false;
        for (const auto value : layer.biases)
            if (!std::isfinite(value))
                return false;
    }
    return true;
}

void testPredictionIgnoresDropout()
{
    const NeuralNetwork plain(config(0.0));
    const NeuralNetwork dropped(config(0.5), plain.layers());
    for (const auto& sample : samples()) {
        test::check(plain.predict(sample.inputs) == dropped.predict(sample.inputs),
            "prediction must use every neuron");
        test::check(dropped.predict(sample.inputs) == dropped.predict(sample.inputs),
            "prediction must be deterministic with dropout");
    }
    test::check(plain.loss(samples()) == dropped.loss(samples()), "loss() must not apply dropout");
}

void testGradientsUseMasks()
{
    for (const auto* activation : {"relu", "sigmoid", "tanh"}) {
        const NeuralNetwork plain(config(0.0, activation));
        test::check(sameGradients(plain.gradients(samples()), plain.gradients(samples())),
            "gradients without dropout must be deterministic");
        const NeuralNetwork dropped(config(0.5, activation), plain.layers());
        const auto first = dropped.gradients(samples());
        const auto second = dropped.gradients(samples());
        test::check(finite(first) && finite(second), "dropout gradients must be finite");
        test::check(!sameGradients(first, second), "dropout must draw new masks for each pass");
        test::check(!sameGradients(first, plain.gradients(samples())),
            "dropout must change the gradients");
    }
}

void testTrainingAndReproducibility()
{
    NeuralNetwork left(config(0.3));
    NeuralNetwork right(config(0.3));
    const auto before = left.loss(samples());
    for (int step = 0; step < 300; ++step) {
        left.trainBatch(samples());
        right.trainBatch(samples());
    }
    test::check(left.loss(samples()) < before, "training with dropout must reduce the loss");
    test::check(sameGradients(left.layers(), right.layers()),
        "the same seed must give the same dropout masks");
}

void testValidationAndSerialization()
{
    test::expectThrow<std::invalid_argument>([] { NeuralNetwork network(config(1.0)); });
    test::expectThrow<std::invalid_argument>([] { NeuralNetwork network(config(-0.1)); });
    test::expectThrow<std::invalid_argument>([] {
        NeuralNetwork network(config(std::numeric_limits<double>::quiet_NaN())); });

    NeuralNetwork network(config(0.25));
    network.setLearningRate(0.01);
    test::check(network.config().learning_rate == 0.01, "learning rate was not updated");
    test::expectThrow<std::invalid_argument>([&] { network.setLearningRate(0.0); });
    test::expectThrow<std::invalid_argument>([&] {
        network.setLearningRate(std::numeric_limits<double>::infinity()); });

    test::TempDirectory directory;
    const auto withDropout = (directory.path / "dropout.nn").string();
    ModelSerializer::save(network, withDropout);
    test::check(ModelSerializer::load(withDropout).config().dropout == 0.25,
        "dropout must survive saving and loading");
    const auto plain = (directory.path / "plain.nn").string();
    ModelSerializer::save(NeuralNetwork(config(0.0)), plain);
    std::ifstream file(plain);
    test::check(!nlohmann::json::parse(file).at("config").contains("dropout"),
        "models without dropout keep the previous format");
    test::check(ModelSerializer::load(plain).config().dropout == 0.0, "missing dropout must mean 0");
}

} // namespace

int main()
{
    testPredictionIgnoresDropout();
    testGradientsUseMasks();
    testTrainingAndReproducibility();
    testValidationAndSerialization();
    std::cout << "Dropout tests passed\n";
    return 0;
}
