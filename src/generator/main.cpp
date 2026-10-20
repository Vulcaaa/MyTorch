#include <charconv>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "neural/NeuralNetwork.hpp"
#include "neural/ModelSerializer.hpp"

namespace {

std::string nextOutputFile(const std::string &baseName, std::size_t &index)
{
    while (true) {
        const auto filename = baseName + "_" + std::to_string(index++) + ".nn";
        if (!std::filesystem::exists(filename) && !std::filesystem::is_symlink(filename))
            return filename;
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "USAGE\n"
            << "    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n"
            << "DESCRIPTION\n"
            << "    config_file_i    Configuration file describing a neural network.\n"
            << "    nb_i             Number of neural networks to generate.\n";
        return 0;
    }
    try {
        if (argc < 3 || (argc - 1) % 2 != 0)
            throw std::invalid_argument("expected pairs of configuration file and count");
        for (int i = 1; i < argc; i += 2) {
            const std::string config_file = argv[i];
            const std::string_view count = argv[i + 1];
            int nb = 0;
            const auto parsed = std::from_chars(count.data(), count.data() + count.size(), nb);
            if (parsed.ec != std::errc{} || parsed.ptr != count.data() + count.size() || nb <= 0)
                throw std::invalid_argument("network count must be a positive integer");

            NetworkConfig config = loadNetworkConfig(config_file);
            const auto baseName = std::filesystem::path(config_file).stem().string();
            std::size_t fileIndex = 1;
            for (int y = 0; y < nb; y++) {
                const auto outputFile = nextOutputFile(baseName, fileIndex);
                NeuralNetwork neuralNetwork(config);
                ModelSerializer::save(neuralNetwork, outputFile);
                ++config.seed;
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "my_torch_generator: " << error.what() << '\n';
        return 84;
    }
    return 0;
}
