#include "neural/NeuralNetwork.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>

int main()
{
    // Non-square layers, every supported activation pair, and a partial second block.
    // predict() is the independent scalar forward path, even in a BLAS build.
    for (const auto* hidden : {"sigmoid", "tanh", "relu", "linear"}) {
        for (const auto* output : {"softmax", "sigmoid"}) {
            NetworkConfig config;
            config.layers = {7, 5, 4, 3};
            config.hidden_activation = hidden;
            config.output_activation = output;
            config.initialization = "xavier";
            config.seed = 43;
            NeuralNetwork network(config);
            std::vector<TrainingSample> samples;
            for (std::size_t s = 0; s < 259; ++s) {
                TrainingSample sample{std::vector<double>(7), std::vector<double>(3, 0.0)};
                for (std::size_t k = 0; k < 7; ++k)
                    sample.inputs[k] = s < 256 && k != s % 7 ? 0.0
                        : std::sin(double(s + 3 * k)) * 0.4;
                sample.target[s % 3] = 1.0;
                samples.push_back(sample);
            }
            auto average = network.gradients({samples.front()});
            for (auto& layer : average) {
                for (auto& row : layer.weights)
                    std::fill(row.begin(), row.end(), 0.0);
                std::fill(layer.biases.begin(), layer.biases.end(), 0.0);
            }
            double expectedLoss = 0.0;
            for (const auto& sample : samples) {
                const auto prediction = network.predict(sample.inputs);
                for (std::size_t k = 0; k < 3; ++k) {
                    expectedLoss -= sample.target[k] * std::log(prediction[k]);
                    if (config.output_activation == "sigmoid")
                        expectedLoss -= (1.0 - sample.target[k]) * std::log1p(-prediction[k]);
                }
                const auto single = network.gradients({sample});
                for (std::size_t i = 0; i < average.size(); ++i)
                    for (std::size_t j = 0; j < average[i].biases.size(); ++j) {
                        average[i].biases[j] += single[i].biases[j] / samples.size();
                        for (std::size_t k = 0; k < average[i].weights[j].size(); ++k)
                            average[i].weights[j][k] += single[i].weights[j][k] / samples.size();
                    }
            }
            test::check(std::abs(network.loss(samples) - expectedLoss / samples.size()) < 1e-12,
                "batched loss differs from scalar predictions across a block boundary");
            const auto batch = network.gradients(samples);
            for (std::size_t i = 0; i < average.size(); ++i)
                for (std::size_t j = 0; j < average[i].biases.size(); ++j) {
                    test::check(std::abs(batch[i].biases[j] - average[i].biases[j]) < 1e-12,
                        "batch bias gradient differs from mean sample gradients");
                    for (std::size_t k = 0; k < average[i].weights[j].size(); ++k)
                        test::check(std::abs(batch[i].weights[j][k] - average[i].weights[j][k]) < 1e-12,
                            "batch weight gradient differs from mean sample gradients");
                }
            const auto before = network.layers();
            const double batchLoss = network.trainBatch(samples);
            test::check(std::abs(batchLoss - expectedLoss / samples.size()) < 1e-12,
                "trainBatch returns the wrong pre-update mean loss");
            for (std::size_t i = 0; i < before.size(); ++i)
                for (std::size_t j = 0; j < before[i].biases.size(); ++j) {
                    test::check(std::abs(network.layers()[i].biases[j]
                        - (before[i].biases[j] - config.learning_rate * average[i].biases[j])) < 1e-12,
                        "batch bias update differs from SGD");
                    for (std::size_t k = 0; k < before[i].weights[j].size(); ++k)
                        test::check(std::abs(network.layers()[i].weights[j][k]
                            - (before[i].weights[j][k] - config.learning_rate * average[i].weights[j][k])) < 1e-12,
                            "batch weight update differs from SGD");
                }
        }
    }
    std::cout << "Batched training tests passed\n";
}
