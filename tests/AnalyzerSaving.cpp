#include "analyzer/Training.hpp"
#include "neural/ModelSerializer.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

namespace {

nlohmann::json readModel(const std::string& path)
{
    std::ifstream stream(path);
    return nlohmann::json::parse(stream);
}

} // namespace

int main()
{
    test::TempDirectory directory;
    const std::string input = (directory.path / "input.nn").string();
    const std::string output = (directory.path / "output.nn").string();
    const std::string reference = (directory.path / "reference.nn").string();

    NetworkConfig config;
    config.layers = {3, 2, 1};
    config.hidden_activation = "relu";
    config.output_activation = "sigmoid";
    config.initialization = "he";
    ModelSerializer::save(NeuralNetwork(config), input);
    const auto original = readModel(input);

    // Use different parameters to ensure saving replaces the original model.
    config.seed = 123;
    NeuralNetwork updated(config);
    ModelSerializer::save(updated, reference);
    const auto expected = readModel(reference);
    test::check(expected != original, "test models must differ");

    AnalyzerArguments arguments{AnalyzerMode::Train, input, "boards.txt", output};
    saveTrainingResult(updated, arguments);
    test::check(readModel(output) == expected, "SAVEFILE did not receive the updated model");
    test::check(readModel(input) == original, "explicit SAVEFILE changed LOADFILE");
    ModelSerializer::load(output);

    arguments.saveFile = input;
    saveTrainingResult(updated, arguments);
    test::check(readModel(input) == expected, "default training destination was not updated");
    ModelSerializer::load(input);

    arguments.saveFile.clear();
    saveTrainingResult(updated, arguments);
    test::check(readModel(input) == expected, "empty SAVEFILE did not fall back to LOADFILE");

    arguments.mode = AnalyzerMode::Predict;
    test::expectThrow<std::invalid_argument>([&] { saveTrainingResult(updated, arguments); });
    test::check(readModel(input) == expected, "prediction changed the model");

    arguments.mode = AnalyzerMode::Train;
    arguments.saveFile = (directory.path / "missing" / "model.nn").string();
    test::expectThrow<std::runtime_error>([&] { saveTrainingResult(updated, arguments); });
    test::check(readModel(input) == expected, "failed save changed LOADFILE");
    std::cout << "AnalyzerSaving tests passed\n";
}
