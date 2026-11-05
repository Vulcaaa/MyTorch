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

double activate(double z, const std::string& activation)
{
    std::vector<double> values{z};
    applyActivation(values, activation);
    return values[0];
}

void testDerivatives()
{
    checkValues(activationDerivative({0.25, 0.5, 0.75}, "sigmoid"), {0.1875, 0.25, 0.1875});
    checkValues(activationDerivative({-0.8, 0, 0.8}, "tanh"), {0.36, 1, 0.36});
    checkValues(activationDerivative({0, 2, 3}, "relu"), {0, 1, 1});
    checkValues(activationDerivative({-2, 0, 3}, "linear"), {1, 1, 1});

    // Slope measured by moving z slightly: (f(z + h) - f(z - h)) / 2h ≈ f'(z).
    // ReLU points stay away from its kink at z = 0.
    constexpr double h = 1e-6;
    for (const auto* activation : {"sigmoid", "tanh", "relu", "linear"}) {
        for (double z : {-2.0, -0.5, 0.3, 1.5}) {
            const double measured = (activate(z + h, activation) - activate(z - h, activation)) / (2 * h);
            const double computed = activationDerivative({activate(z, activation)}, activation)[0];
            test::check(std::abs(measured - computed) < 1e-7,
                std::string("derivative does not match the measured slope for ") + activation);
        }
    }

    for (const auto* activation : {"relu", "linear", "sigmoid", "tanh"})
        test::check(activationDerivative({}, activation).empty(), "derivative changed vector size");
    test::expectThrow<std::invalid_argument>([] { activationDerivative({0.5}, "softmax"); });
    test::expectThrow<std::invalid_argument>([] { activationDerivative({0.5}, "unknown"); });
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
    testDerivatives();
    std::cout << "Activation tests passed\n";
}
