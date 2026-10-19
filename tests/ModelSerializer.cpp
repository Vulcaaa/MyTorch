#include "mytorch/ModelSerializer.hpp"

#include <nlohmann/json.hpp>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Json = nlohmann::json;

namespace {

Json readJson(const std::filesystem::path &path)
{
    std::ifstream file(path);
    return Json::parse(file);
}

void writeJson(const std::filesystem::path &path, const Json &model)
{
    std::ofstream file(path);
    file << model.dump(2);
}

void expectInvalid(const std::filesystem::path &path, const Json &model)
{
    writeJson(path, model);
    try {
        ModelSerializer::load(path.string());
    } catch (const std::runtime_error &error) {
        assert(std::string(error.what()).find(path.string()) != std::string::npos);
        return;
    }
    throw std::runtime_error("invalid model was accepted");
}

void runTests(const std::filesystem::path &directory)
{
    const auto source = directory / "source.nn";
    const auto restored = directory / "restored.nn";
    NetworkConfig config;
    config.layers = {3, 2, 1};
    config.hidden_activation = "relu";
    config.output_activation = "sigmoid";
    config.initialization = "he";
    ModelSerializer::save(NeuralNetwork(config), source.string());
    const auto generated = readJson(source);
    assert(generated.at("layers").size() == 2);
    assert(generated.at("layers")[0].at("weights").size() == 2);
    assert(generated.at("layers")[0].at("weights")[0].size() == 3);

    // Values different from initialization must survive loading exactly.
    auto model = generated;
    model["layers"][0]["weights"][0][0] = 1.2345678901234567;
    model["layers"][0]["biases"][0] = -0.125;
    model["config"]["learning_rate"] = 0.025;
    model["config"]["seed"] = 123;
    writeJson(source, model);
    ModelSerializer::save(ModelSerializer::load(source.string()), restored.string());
    assert(readJson(restored) == model);

    auto invalid = model;
    invalid["version"] = 2;
    expectInvalid(source, invalid);
    invalid["version"] = 1.0;
    expectInvalid(source, invalid);
    invalid = model;
    invalid.erase("config");
    expectInvalid(source, invalid);
    invalid = model;
    invalid["config"]["layers"] = {3};
    expectInvalid(source, invalid);
    invalid["config"]["layers"] = {3, 0, 1};
    expectInvalid(source, invalid);
    invalid["config"]["layers"] = {3, -2, 1};
    expectInvalid(source, invalid);
    invalid["config"]["layers"] = {3, 2.5, 1};
    expectInvalid(source, invalid);
    invalid = model;
    invalid["config"]["seed"] = 4294967296ULL;
    expectInvalid(source, invalid);
    invalid = model;
    invalid["config"]["hidden_activation"] = "unknown";
    expectInvalid(source, invalid);
    invalid = model;
    invalid["config"]["learning_rate"] = true;
    expectInvalid(source, invalid);
    invalid = model;
    invalid["layers"].erase(1);
    expectInvalid(source, invalid);
    invalid = model;
    invalid["layers"][0]["weights"][0].erase(0);
    expectInvalid(source, invalid);
    invalid = model;
    invalid["layers"][0]["biases"] = Json::array();
    expectInvalid(source, invalid);
    for (const auto &value : Json::array({true, nullptr, "invalid"})) {
        invalid = model;
        invalid["layers"][0]["weights"][0][0] = value;
        expectInvalid(source, invalid);
    }

    for (const auto *content : {"{", "{} trailing", "{\"version\":1e999}"}) {
        {
            std::ofstream file(source);
            file << content;
        }
        bool rejected = false;
        try {
            ModelSerializer::load(source.string());
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        assert(rejected);
    }
    bool rejected = false;
    try {
        ModelSerializer::load((directory / "missing.nn").string());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);
    rejected = false;
    try {
        ModelSerializer::save(NeuralNetwork(config), directory.string());
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main()
{
    const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path()
        / ("mytorch-serializer-" + std::to_string(id));
    std::filesystem::create_directory(directory);
    try {
        runTests(directory);
    } catch (...) {
        std::filesystem::remove_all(directory);
        throw;
    }
    std::filesystem::remove_all(directory);
    std::cout << "ModelSerializer tests passed\n";
}
