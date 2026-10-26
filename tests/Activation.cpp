#include "neural/Activation.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <iostream>
#include <numeric>

namespace {

void checkValues(const std::vector<double>& actual, const std::vector<double>& expected)
{
    test::check(actual.size() == expected.size(), "activation changed vector size");
    for (std::size_t i = 0; i < actual.size(); ++i)
        test::check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 1e-12,
            "incorrect activation value at index " + std::to_string(i));
}

} // namespace

int main()
{
    std::vector<double> values{-2, 0, 3};
    applyActivation(values, "relu");
    checkValues(values, {0, 0, 3});
    values = {-2, 0, 3};
    applyActivation(values, "linear");
    checkValues(values, {-2, 0, 3});
    values = {-std::log(3.0), 0, std::log(3.0)};
    applyActivation(values, "sigmoid");
    checkValues(values, {0.25, 0.5, 0.75});
    values = {-1000, 1000};
    applyActivation(values, "sigmoid");
    checkValues(values, {0, 1});
    values = {-std::log(3.0), 0, std::log(3.0)};
    applyActivation(values, "tanh");
    checkValues(values, {-0.8, 0, 0.8});

    // Ratios exp(0):exp(log(2)):exp(log(3)) = 1:2:3.
    for (double offset : {0.0, 1000.0, -1000.0}) {
        values = {offset, offset + std::log(2.0), offset + std::log(3.0)};
        applyActivation(values, "softmax");
        checkValues(values, {1.0 / 6, 2.0 / 6, 3.0 / 6});
        test::check(std::abs(std::accumulate(values.begin(), values.end(), 0.0) - 1) < 1e-12,
            "softmax probabilities do not sum to one");
    }
    values = {-1000, 0, 1000};
    applyActivation(values, "softmax");
    checkValues(values, {0, 0, 1});
    values = {1000, 1000, 1000};
    applyActivation(values, "softmax");
    checkValues(values, {1.0 / 3, 1.0 / 3, 1.0 / 3});
    values = {42};
    applyActivation(values, "softmax");
    checkValues(values, {1});
    for (const auto* activation : {"relu", "linear", "sigmoid", "tanh", "softmax"}) {
        values.clear();
        applyActivation(values, activation);
        checkValues(values, {});
    }
    for (const auto& inputs : std::vector<std::vector<double>>{{}, {1, 2}}) {
        values = inputs;
        test::expectThrow<std::invalid_argument>([&] { applyActivation(values, "unknown"); });
    }
    std::cout << "Activation tests passed\n";
}
