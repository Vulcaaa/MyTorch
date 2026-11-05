#include <charconv>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "logging/Logger.hpp"
#include "neural/NeuralNetwork.hpp"
#include "neural/ModelSerializer.hpp"

namespace {

std::string nextOutputFile(const std::string &baseName, std::size_t &index)
{
    while (true) {
        const auto filename = std::filesystem::path("models") /
            (baseName + "_" + std::to_string(index++) + ".nn");
        if (!std::filesystem::exists(filename) && !std::filesystem::is_symlink(filename))
            return filename.string();
    }
}

std::string describe(const NetworkConfig &config)
{
    std::ostringstream description;
    description << "layers=";
    for (std::size_t i = 0; i < config.layers.size(); ++i)
        description << (i == 0 ? "" : ",") << config.layers[i];
    description << " hidden_activation=" << config.hidden_activation
        << " output_activation=" << config.output_activation
        << " initialization=" << config.initialization
        << " learning_rate=" << config.learning_rate
        << " seed=" << config.seed;
    return description.str();
}

std::string elapsedSince(std::chrono::steady_clock::time_point start)
{
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    std::ostringstream text;
    text << std::fixed << std::setprecision(3) << elapsed.count() << " s";
    return text.str();
}

// Generates every requested model and returns how many were saved.
std::size_t generateModels(int ac, char **av, Logger &logger)
{
    if (ac < 3 || (ac - 1) % 2 != 0)
        throw std::invalid_argument("expected pairs of configuration file and count");
    std::size_t generated = 0;
    for (int i = 1; i < ac; i += 2) {
        const std::string config_file = av[i];
        const std::string_view count = av[i + 1];
        int nb = 0;
        const auto parsed = std::from_chars(count.data(), count.data() + count.size(), nb);
        if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size() || nb <= 0)
            throw std::invalid_argument("network count must be a positive integer");

        NetworkConfig config = loadNetworkConfig(config_file);
        logger.info("Configuration " + config_file + ": " + describe(config));
        std::filesystem::create_directories("models");
        const auto baseName = std::filesystem::path(config_file).stem().string();
        std::size_t fileIndex = 1;
        for (int y = 0; y < nb; y++) {
            const auto outputFile = nextOutputFile(baseName, fileIndex);
            NeuralNetwork neuralNetwork(config);
            ModelSerializer::save(neuralNetwork, outputFile);
            logger.info("Generated " + outputFile + " (seed " + std::to_string(config.seed) + ")");
            ++config.seed;
            ++generated;
        }
    }
    return generated;
}

} // namespace

int main(int ac, char **av)
{
    if (ac == 2 && std::string_view(av[1]) == "--help") {
        std::cout << "USAGE\n"
            << "    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n"
            << "DESCRIPTION\n"
            << "    config_file_i    Configuration file describing a neural network.\n"
            << "    nb_i             Number of neural networks to generate.\n"
            << "    Models are saved in models/ under the current directory.\n"
            << "    Each run is logged in logs/generation_<UTC time>.log.\n";
        return 0;
    }
    const auto start = std::chrono::steady_clock::now();
    Logger logger("logs", "generation", std::cerr);
    logger.info("Generation started");
    try {
        const std::size_t generated = generateModels(ac, av, logger);
        logger.info("Generation complete: " + std::to_string(generated)
            + (generated == 1 ? " model in " : " models in ") + elapsedSince(start));
    } catch (const std::exception &error) {
        logger.error("Generation failed after " + elapsedSince(start) + ": " + error.what());
        std::cerr << "my_torch_generator: " << error.what() << '\n';
        return 84;
    }
    return 0;
}
