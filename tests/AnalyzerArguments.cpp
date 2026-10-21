#include "analyzer/Arguments.hpp"
#include "TestSupport.hpp"

#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

AnalyzerArguments parse(std::initializer_list<const char*> arguments)
{
    std::vector<const char*> argv = {"my_torch_analyzer"};
    argv.insert(argv.end(), arguments.begin(), arguments.end());
    return parseAnalyzerArguments(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main()
{
    auto arguments = parse({"--predict", "model.nn", "positions.txt"});
    test::check(arguments.mode == AnalyzerMode::Predict, "incorrect prediction mode");
    test::check(arguments.loadFile == "model.nn" && arguments.chessFile == "positions.txt",
        "incorrect input paths");
    test::check(arguments.saveFile.empty(), "prediction must not save the model");

    arguments = parse({"--train", "model.nn", "positions.txt"});
    test::check(arguments.mode == AnalyzerMode::Train, "incorrect training mode");
    test::check(arguments.saveFile == "model.nn", "training must default to saving in LOADFILE");

    arguments = parse({"--train", "--save", "trained model.nn", "initial model.nn", "chess boards.txt"});
    test::check(arguments.mode == AnalyzerMode::Train, "--save changed the mode");
    test::check(arguments.saveFile == "trained model.nn" && arguments.loadFile == "initial model.nn"
        && arguments.chessFile == "chess boards.txt", "incorrect paths with --save");

    const std::vector<std::vector<const char*>> invalid = {
        {}, {"--train"}, {"--train", "model.nn"},
        {"--unknown", "model.nn", "positions.txt"},
        {"--predict", "--save", "out.nn", "model.nn", "positions.txt"},
        {"--train", "--save", "model.nn", "positions.txt"},
        {"--train", "--save", "out.nn", "model.nn"},
        {"--train", "model.nn", "positions.txt", "extra"},
        {"--train", "--save", "out.nn", "model.nn", "positions.txt", "extra"},
        {"--train", "--save", "", "model.nn", "positions.txt"},
        {"--train", "", "positions.txt"}, {"--predict", "model.nn", ""},
        {"--train", "--unknown", "positions.txt"},
        {"--train", "model.nn", "--unknown"},
        {"--train", "--save", "--unknown", "model.nn", "positions.txt"},
        {"--predict", "--train", "model.nn", "positions.txt"}
    };
    for (auto argv : invalid) {
        argv.insert(argv.begin(), "my_torch_analyzer");
        const auto error = test::expectThrow<std::invalid_argument>([&] {
            parseAnalyzerArguments(static_cast<int>(argv.size()), argv.data());
        });
        test::check(!error.empty(), "invalid arguments need a diagnostic");
    }
    std::cout << "AnalyzerArguments tests passed\n";
}
