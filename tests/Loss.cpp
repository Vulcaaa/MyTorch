#include "neural/Activation.hpp"
#include "neural/Loss.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

void checkClose(double actual, double expected, const std::string& message)
{
    test::check(std::isfinite(actual) && std::abs(actual - expected) < 1e-12, message);
}

// Textbook formula from probabilities, with 0 * log(0) taken as 0.
// It breaks when the probability of an expected answer rounds to 0.
double textbookLoss(std::vector<double> logits, const std::vector<double>& target,
                    const std::string& activation)
{
    applyActivation(logits, activation);
    double loss = 0.0;
    for (std::size_t k = 0; k < logits.size(); ++k) {
        if (target[k] > 0.0)
            loss -= target[k] * std::log(logits[k]);
        if (activation == "sigmoid" && target[k] < 1.0)
            loss -= (1.0 - target[k]) * std::log(1.0 - logits[k]);
    }
    return loss;
}

void testCrossEntropy()
{
    // softmax([log 1, log 2, log 3]) = [1/6, 2/6, 3/6]: the expected class gets 1/3.
    std::vector<double> logits{0.0, std::log(2.0), std::log(3.0)};
    checkClose(sampleLoss(logits, {0, 1, 0}, "softmax"), std::log(3.0), "cross-entropy value");
    // Adding the same value to every logit leaves softmax, and the loss, unchanged.
    for (double offset : {1000.0, -1000.0}) {
        std::vector<double> shifted{offset, offset + std::log(2.0), offset + std::log(3.0)};
        checkClose(sampleLoss(shifted, {0, 1, 0}, "softmax"), std::log(3.0),
            "cross-entropy must not overflow with large logits");
    }
    checkClose(sampleLoss({0, 0}, {0.5, 0.5}, "softmax"), std::log(2.0), "soft target");

    // The expected class gets a probability that rounds to 0: the textbook loss is infinite,
    // the stable one is the logit gap.
    test::check(std::isinf(textbookLoss({800, 0, 0}, {0, 0, 1}, "softmax")),
        "textbook cross-entropy was expected to overflow");
    checkClose(sampleLoss({800, 0, 0}, {0, 0, 1}, "softmax"), 800, "cross-entropy far from target");
    const double confident = sampleLoss({100, 0, 0}, {1, 0, 0}, "softmax");
    test::check(confident >= 0 && confident < 1e-12, "confident correct answer must cost ~0");
}

void testBinaryCrossEntropy()
{
    // sigmoid(0) = 1/2 and sigmoid(log 3) = 3/4.
    checkClose(sampleLoss({0}, {1}, "sigmoid"), std::log(2.0), "binary cross-entropy at 1/2");
    checkClose(sampleLoss({std::log(3.0)}, {1}, "sigmoid"), -std::log(0.75), "target 1");
    checkClose(sampleLoss({std::log(3.0)}, {0}, "sigmoid"), -std::log(0.25), "target 0");
    // Outputs are independent, so their losses add up: log 2 + log 4 = log 8.
    checkClose(sampleLoss({0, std::log(3.0)}, {1, 0}, "sigmoid"), std::log(8.0), "summed outputs");

    // sigmoid(40) rounds to exactly 1, so the textbook loss computes log(0).
    test::check(std::isinf(textbookLoss({40}, {0}, "sigmoid")),
        "textbook binary cross-entropy was expected to overflow");
    checkClose(sampleLoss({40}, {0}, "sigmoid"), 40, "binary cross-entropy far from target 0");
    checkClose(sampleLoss({-800}, {1}, "sigmoid"), 800, "binary cross-entropy far from target 1");
}

void testMatchesTextbookFormula()
{
    const std::vector<double> logits{-1.3, 0.4, 2.1};
    for (const auto& target : std::vector<std::vector<double>>{{1, 0, 0}, {0, 0, 1}, {0.2, 0.3, 0.5}})
        checkClose(sampleLoss(logits, target, "softmax"), textbookLoss(logits, target, "softmax"),
            "cross-entropy differs from the textbook formula");
    for (const auto& target : std::vector<std::vector<double>>{{1, 0, 1}, {0.3, 0.9, 0}})
        checkClose(sampleLoss(logits, target, "sigmoid"), textbookLoss(logits, target, "sigmoid"),
            "binary cross-entropy differs from the textbook formula");
}

void testInvalidArguments()
{
    for (const auto* activation : {"relu", "tanh", "linear", "unknown"})
        test::expectThrow<std::invalid_argument>([&] { sampleLoss({0, 0}, {1, 0}, activation); });
    test::expectThrow<std::invalid_argument>([] { sampleLoss({0, 0}, {1}, "softmax"); });
    test::expectThrow<std::invalid_argument>([] { sampleLoss({}, {}, "sigmoid"); });
}

} // namespace

int main()
{
    testCrossEntropy();
    testBinaryCrossEntropy();
    testMatchesTextbookFormula();
    testInvalidArguments();
    std::cout << "Loss tests passed\n";
}
