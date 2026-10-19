#include "mytorch/Initializer.hpp"
#include "mytorch/NeuralNetwork.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

int main()
{
    using initialization::Method;
    test::check(initialization::parseMethod("xavier") == Method::Xavier, "Xavier not recognized");
    test::check(initialization::parseMethod("he") == Method::He, "He not recognized");
    test::expectThrow<std::invalid_argument>([] { initialization::parseMethod("unknown"); });

    for (const auto method : {Method::Xavier, Method::He}) {
        Layer layer(64, 512);
        std::fill(layer.biases.begin(), layer.biases.end(), 7.0);
        std::mt19937 rng(42);
        initialization::initialize(layer, method, rng);
        Layer repeated(64, 512);
        std::mt19937 sameSeed(42);
        initialization::initialize(repeated, method, sameSeed);
        test::check(layer.weights == repeated.weights, "same seed is not reproducible");
        initialization::initialize(repeated, method, sameSeed);
        test::check(layer.weights != repeated.weights, "RNG state did not advance");
        test::check(layer.weights.size() == 512 && layer.biases.size() == 512,
            "initialization changed dimensions");
        double sum = 0.0;
        double sumSquares = 0.0;
        for (const auto &row : layer.weights) {
            test::check(row.size() == 64, "initialization changed row size");
            for (const auto weight : row) {
                test::check(std::isfinite(weight), "non-finite weight");
                if (method == Method::Xavier)
                    test::check(std::abs(weight) <= std::sqrt(6.0 / (64 + 512)),
                        "Xavier weight outside expected bounds");
                sum += weight;
                sumSquares += weight * weight;
            }
        }
        for (const auto bias : layer.biases)
            test::check(bias == 0.0, "bias was not reset");
        const double mean = sum / (64 * 512);
        const double variance = sumSquares / (64 * 512) - mean * mean;
        const double expected = method == Method::Xavier ? 2.0 / (64 + 512) : 2.0 / 64;
        // Broad statistical bounds, with a fixed seed and 32768 samples.
        test::check(std::abs(mean) < std::sqrt(expected) * 0.03, "weight mean is too far from zero");
        test::check(std::abs(variance / expected - 1.0) < 0.1, "incorrect initialization variance");
    }
    std::mt19937 rng(42);
    for (auto invalid : {Layer(0, 1), Layer(1, 0)})
        test::expectThrow<std::invalid_argument>([&] {
            initialization::initialize(invalid, Method::He, rng);
        });
    Layer invalid(2, 2);
    invalid.weights[1].pop_back();
    test::expectThrow<std::invalid_argument>([&] {
        initialization::initialize(invalid, Method::He, rng);
    });
    invalid = Layer(2, 2);
    invalid.biases.pop_back();
    test::expectThrow<std::invalid_argument>([&] {
        initialization::initialize(invalid, Method::He, rng);
    });
    Layer valid(1, 1);
    test::expectThrow<std::invalid_argument>([&] {
        initialization::initialize(valid, static_cast<Method>(-1), rng);
    });
    std::cout << "Initializer tests passed\n";
}
