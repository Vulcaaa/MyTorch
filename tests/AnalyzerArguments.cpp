#include "analyzer/Arguments.hpp"
#include "TestSupport.hpp"

#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

AnalyzerArguments parse(const std::vector<const char*>& arguments)
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

    arguments = parse({"--train", "--validation", "validation.txt", "model.nn", "positions.txt"});
    test::check(arguments.mode == AnalyzerMode::Train, "incorrect training mode");
    test::check(arguments.saveFile == "model.nn", "training must default to saving in LOADFILE");
    test::check(arguments.training.validationFile == "validation.txt", "incorrect validation path");
    test::check(arguments.training.historyFile.empty(), "history path must default to automatic");
    test::check(arguments.training.epochs == 100 && arguments.training.batchSize == 32
        && arguments.training.patience == 10 && arguments.training.minDelta == 0.0001,
        "incorrect training defaults");

    arguments = parse({"--train", "--history", "run history.csv", "--batch-size", "16",
        "--save", "trained model.nn", "--min-delta", "1e-3", "--epochs", "250",
        "--validation", "validation data.txt", "--patience", "20",
        "initial model.nn", "chess boards.txt"});
    test::check(arguments.mode == AnalyzerMode::Train, "--save changed the mode");
    test::check(arguments.saveFile == "trained model.nn" && arguments.loadFile == "initial model.nn"
        && arguments.chessFile == "chess boards.txt", "incorrect paths with --save");
    test::check(arguments.training.historyFile == "run history.csv"
        && arguments.training.validationFile == "validation data.txt", "incorrect auxiliary paths");
    test::check(arguments.training.epochs == 250 && arguments.training.batchSize == 16
        && arguments.training.patience == 20 && arguments.training.minDelta == 0.001,
        "training overrides were not applied");
    arguments = parse({"--train", "--validation", "val.txt", "--min-delta", "0",
        "--epochs", "1", "--batch-size", "1", "--patience", "1", "model.nn", "train.txt"});
    test::check(arguments.training.minDelta == 0 && arguments.training.epochs == 1
        && arguments.training.batchSize == 1 && arguments.training.patience == 1,
        "valid lower bounds were rejected");
    const auto maximum = std::to_string(std::numeric_limits<std::size_t>::max());
    arguments = parse({"--train", "--validation", "val.txt", "--epochs", maximum.c_str(),
        "model.nn", "train.txt"});
    test::check(arguments.training.epochs == std::numeric_limits<std::size_t>::max(),
        "maximum representable integer was rejected");

    const std::vector<std::vector<const char*>> invalid = {
        {}, {"--train"}, {"--train", "model.nn"}, {"--train", "model.nn", "positions.txt"},
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
    for (const auto& argv : invalid) {
        const auto error = test::expectThrow<std::invalid_argument>([&] {
            parse(argv);
        });
        test::check(!error.empty(), "invalid arguments need a diagnostic");
    }

    for (const auto* option : {"--validation", "--save", "--history", "--epochs",
        "--batch-size", "--patience", "--min-delta"}) {
        const auto* value = std::string(option) == "--min-delta" ? "0.01"
            : std::string(option) == "--epochs" || std::string(option) == "--batch-size"
                || std::string(option) == "--patience" ? "1" : "file";
        const std::vector<std::vector<const char*>> cases = {
            {"--predict", option, value, "model.nn", "train.txt"},
            {"--train", option},
            {"--train", option, "", "model.nn", "train.txt"},
            {"--train", option, "--validation", "val.txt", "model.nn", "train.txt"},
            {"--train", option, value, option, value, "model.nn", "train.txt"},
            {"--train", "--validation", "val.txt", "model.nn", "train.txt", option, value}
        };
        for (const auto& argv : cases) {
            test::expectThrow<std::invalid_argument>([&] { parse(argv); });
        }
    }
    for (const auto* option : {"--epochs", "--batch-size", "--patience", "--min-delta"}) {
        std::vector<const char*> values = {"-1", "+1", "1abc", " 1", "1 ", "nan", "inf",
            "1e9999", "0x10", "9999999999999999999999999999999999999999"};
        if (std::string(option) == "--min-delta") {
            values.pop_back(); // Large finite doubles are valid, unlike overflowing integers.
            values.push_back("1e-9999");
        } else {
            values.insert(values.end(), {"0", "1.5", "1e2"});
        }
        for (const auto* value : values) {
            const auto error = test::expectThrow<std::invalid_argument>([&] {
                parse({"--train", "--validation", "val.txt", option, value, "model.nn", "train.txt"});
            });
            test::check(error.find(option) != std::string::npos, "numeric error must identify option");
        }
    }
    std::cout << "AnalyzerArguments tests passed\n";
}
