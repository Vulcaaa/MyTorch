#include "neural/NeuralNetwork.hpp"
#include "TestSupport.hpp"

#include <iostream>
#include <numeric>

namespace {
void sameLayers(const std::vector<Layer>& a, const std::vector<Layer>& b)
{
    test::check(a.size() == b.size(), "layer count differs");
    for (std::size_t i = 0; i < a.size(); ++i)
        test::check(a[i].weights == b[i].weights && a[i].biases == b[i].biases,
            "binary storage changed numeric results");
}
}

int main()
{
    for (const auto* optimizer : {"sgd", "adam"}) {
        for (const auto* hidden : {"sigmoid", "tanh", "relu"}) {
            NetworkConfig config;
            config.layers = {7, 5, 3};
            config.hidden_activation = hidden;
            config.output_activation = "softmax";
            config.initialization = "xavier";
            config.optimizer = optimizer;
            NeuralNetwork dense(config), compact(config);
            std::vector<TrainingSample> originals, packed;
            for (std::size_t s = 0; s < 259; ++s) {
                TrainingSample sample{std::vector<double>(7), std::vector<double>(3)};
                sample.target[s % 3] = 1.0;
                TrainingSample binary{{}, sample.target};
                binary.binaryInputSize = 7;
                for (std::size_t k = 0; k < 7; ++k) {
                    if ((s + k) % 3 == 0 || s > 255) {
                        sample.inputs[k] = 1.0;
                        binary.activeInputs.push_back(k);
                    }
                }
                test::check(binary.expandInputs() == sample.inputs, "binary expansion changed inputs");
                originals.push_back(sample);
                packed.push_back(binary);
            }
            test::check(dense.loss(originals) == compact.loss(packed), "binary loss changed");
            sameLayers(dense.gradients(originals), compact.gradients(packed));
            test::check(dense.trainBatch(originals) == compact.trainBatch(packed), "batch loss changed");
            sameLayers(dense.layers(), compact.layers());
            std::vector<std::size_t> order(originals.size());
            std::iota(order.rbegin(), order.rend(), 0);
            dense.trainBatch(originals, order, 1, 128);
            compact.trainBatch(packed, order, 1, 128);
            sameLayers(dense.layers(), compact.layers());
            auto invalid = packed.front();
            for (auto indices : {std::vector<std::size_t>{7}, {1, 1}, {2, 1}}) {
                invalid.activeInputs = indices;
                test::expectThrow<std::invalid_argument>([&] { compact.trainBatch({invalid}); });
            }
            invalid = packed.front();
            invalid.inputs = originals.front().inputs;
            test::expectThrow<std::invalid_argument>([&] { compact.trainBatch({invalid}); });
            invalid = packed.front();
            invalid.binaryInputSize = 0;
            test::expectThrow<std::invalid_argument>([&] { compact.trainBatch({invalid}); });
        }
    }
    std::cout << "Binary training tests passed\n";
}
