#include "mytorch/ModelSerializer.hpp"

#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using Json = nlohmann::json;
constexpr unsigned int formatVersion = 1;

void require(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template<typename T>
T readUnsigned(const Json &value, const std::string &name)
{
    require(value.is_number_unsigned()
        || (value.is_number_integer() && value.get<std::int64_t>() >= 0),
        name + " must be a non-negative integer");
    const auto number = value.get<std::uint64_t>();
    require(number <= std::numeric_limits<T>::max(), name + " is too large");
    return static_cast<T>(number);
}

void validateConfig(const NetworkConfig &config)
{
    require(config.layers.size() >= 2, "a network requires at least two layers");
    for (const auto size : config.layers)
        require(size > 0, "layer sizes must be positive");
    require(config.hidden_activation == "relu" || config.hidden_activation == "sigmoid"
        || config.hidden_activation == "tanh", "invalid hidden_activation");
    require(config.output_activation == "sigmoid" || config.output_activation == "softmax",
        "invalid output_activation");
    require(std::isfinite(config.learning_rate) && config.learning_rate > 0.0,
        "learning_rate must be finite and positive");
    require(config.initialization == "he" || config.initialization == "xavier",
        "invalid initialization");
}

NetworkConfig readConfig(const Json &data)
{
    NetworkConfig config;
    const auto &sizes = data.at("layers");
    require(sizes.is_array(), "config.layers must be an array");
    for (const auto &size : sizes)
        config.layers.push_back(readUnsigned<std::size_t>(size, "layer size"));
    config.hidden_activation = data.at("hidden_activation").get<std::string>();
    config.output_activation = data.at("output_activation").get<std::string>();
    require(data.at("learning_rate").is_number(), "learning_rate must be a number");
    config.learning_rate = data.at("learning_rate").get<double>();
    config.initialization = data.at("initialization").get<std::string>();
    config.seed = readUnsigned<std::uint32_t>(data.at("seed"), "seed");
    validateConfig(config);
    return config;
}

void validateLayer(const Layer &layer, std::size_t inputs, std::size_t neurons)
{
    require(layer.weights.size() == neurons, "incorrect number of weight rows");
    require(layer.biases.size() == neurons, "incorrect number of biases");
    for (const auto &row : layer.weights) {
        require(row.size() == inputs, "incorrect number of weights per neuron");
        for (const auto weight : row)
            require(std::isfinite(weight), "weights must be finite");
    }
    for (const auto bias : layer.biases)
        require(std::isfinite(bias), "biases must be finite");
}

std::vector<Layer> readLayers(const Json &data, const NetworkConfig &config)
{
    require(data.is_array() && data.size() == config.layers.size() - 1,
        "incorrect number of parameter layers");
    std::vector<Layer> layers;
    layers.reserve(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        const auto &weights = data[i].at("weights");
        const auto &biases = data[i].at("biases");
        const auto inputs = config.layers[i];
        const auto neurons = config.layers[i + 1];
        // Check dimensions before allocating from values declared in the config.
        require(weights.is_array() && weights.size() == neurons,
            "incorrect number of weight rows");
        require(biases.is_array() && biases.size() == neurons,
            "incorrect number of biases");
        for (const auto &row : weights) {
            require(row.is_array() && row.size() == inputs,
                "incorrect number of weights per neuron");
            for (const auto &weight : row)
                require(weight.is_number(), "weights must be numbers");
        }
        for (const auto &bias : biases)
            require(bias.is_number(), "biases must be numbers");
        Layer layer(inputs, neurons);
        layer.weights = weights.get<std::vector<std::vector<double>>>();
        layer.biases = biases.get<std::vector<double>>();
        validateLayer(layer, inputs, neurons);
        layers.push_back(std::move(layer));
    }
    return layers;
}

Json writeConfig(const NetworkConfig &config)
{
    return {
        {"layers", config.layers},
        {"hidden_activation", config.hidden_activation},
        {"output_activation", config.output_activation},
        {"learning_rate", config.learning_rate},
        {"initialization", config.initialization},
        {"seed", config.seed}
    };
}

} // namespace

void ModelSerializer::save(const NeuralNetwork &network, const std::string &filepath)
{
    try {
        validateConfig(network._config);
        require(network._layers.size() == network._config.layers.size() - 1,
            "incorrect number of parameter layers");
        Json layers = Json::array();
        for (std::size_t i = 0; i < network._layers.size(); ++i) {
            const auto &layer = network._layers[i];
            validateLayer(layer, network._config.layers[i], network._config.layers[i + 1]);
            layers.push_back({{"weights", layer.weights}, {"biases", layer.biases}});
        }
        const Json model = {
            {"format", "mytorch"},
            {"version", formatVersion},
            {"config", writeConfig(network._config)},
            {"layers", std::move(layers)}
        };
        // Serialize before opening the file, so invalid data cannot truncate it.
        const auto content = model.dump(2);
        std::ofstream file(filepath);
        require(file.is_open(), "cannot open model for writing");
        file << content << '\n';
        file.close();
        require(!file.fail(), "error writing model");
    } catch (const Json::exception &error) {
        throw std::runtime_error(filepath + ": invalid model: " + error.what());
    } catch (const std::runtime_error &error) {
        throw std::runtime_error(filepath + ": " + error.what());
    }
}

NeuralNetwork ModelSerializer::load(const std::string &filepath)
{
    try {
        std::ifstream file(filepath);
        require(file.is_open(), "cannot open model for reading");
        const auto model = Json::parse(file);
        require(!file.bad(), "error reading model");
        require(model.at("format") == "mytorch", "unknown model format");
        require(readUnsigned<unsigned int>(model.at("version"), "version") == formatVersion,
            "unsupported model version");
        auto config = readConfig(model.at("config"));
        auto layers = readLayers(model.at("layers"), config);
        return NeuralNetwork(std::move(config), std::move(layers));
    } catch (const Json::exception &error) {
        throw std::runtime_error(filepath + ": invalid model JSON: " + error.what());
    } catch (const std::runtime_error &error) {
        throw std::runtime_error(filepath + ": " + error.what());
    }
}
