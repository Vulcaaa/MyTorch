#include "NetworkConfig.hpp"

#include <yaml-cpp/yaml.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {

template<typename T>
T parseUnsigned(std::string_view value)
{
    T result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::runtime_error("expected a non-negative integer");
    return result;
}

std::vector<std::size_t> parseLayers(const YAML::Node &value)
{
    if (!value.IsSequence() || value.size() < 2)
        throw std::runtime_error("layers requires a YAML list with at least an input and output layer");
    std::vector<std::size_t> layers;
    for (const auto &item : value) {
        if (!item.IsScalar())
            throw std::runtime_error("layer sizes must be positive integers");
        const auto size = parseUnsigned<std::size_t>(item.Scalar());
        if (size == 0)
            throw std::runtime_error("layer sizes must be positive");
        layers.push_back(size);
    }
    return layers;
}

double parseLearningRate(std::string_view value)
{
    std::size_t consumed = 0;
    double rate;
    try {
        rate = std::stod(std::string(value), &consumed);
    } catch (const std::exception &) {
        throw std::runtime_error("learning_rate must be a finite positive number");
    }
    if (consumed != value.size() || !std::isfinite(rate) || rate <= 0.0)
        throw std::runtime_error("learning_rate must be a finite positive number");
    return rate;
}

void setOption(NetworkConfig &options, const std::string &key, const YAML::Node &node)
{
    if (key == "layers") {
        options.layers = parseLayers(node);
        return;
    }
    if (!node.IsScalar())
        throw std::runtime_error(key + " must be a scalar value");
    const std::string value = node.Scalar();
    if (key == "hidden_activation") {
        if (value != "sigmoid" && value != "relu" && value != "tanh")
            throw std::runtime_error("hidden_activation must be sigmoid, relu or tanh");
        options.hidden_activation = value;
    } else if (key == "output_activation") {
        if (value != "sigmoid" && value != "softmax")
            throw std::runtime_error("output_activation must be sigmoid or softmax");
        options.output_activation = value;
    } else if (key == "learning_rate") {
        options.learning_rate = parseLearningRate(value);
    } else if (key == "optimizer") {
        if (value != "sgd" && value != "adam")
            throw std::runtime_error("optimizer must be sgd or adam");
        options.optimizer = value;
    } else if (key == "initialization") {
        if (value != "xavier" && value != "he")
            throw std::runtime_error("initialization must be xavier or he");
        options.initialization = value;
    } else if (key == "seed") {
        options.seed = parseUnsigned<std::uint32_t>(value);
    } else {
        throw std::runtime_error("unknown key '" + key + "'");
    }
}

} // namespace

NetworkConfig loadNetworkConfig(const std::string &filepath)
{
    std::ifstream file(filepath);
    if (!file)
        throw std::runtime_error(filepath + ": cannot open configuration file");
    NetworkConfig options;
    std::set<std::string> keys;
    try {
        const auto documents = YAML::LoadAll(file);
        if (documents.size() != 1 || !documents.front().IsMap())
            throw std::runtime_error(filepath + ": expected one YAML mapping");
        for (const auto &entry : documents.front()) {
            const auto line = std::to_string(entry.first.Mark().line + 1);
            try {
                if (!entry.first.IsScalar())
                    throw std::runtime_error("configuration keys must be strings");
                const auto key = entry.first.Scalar();
                if (!keys.insert(key).second)
                    throw std::runtime_error("duplicate key '" + key + "'");
                setOption(options, key, entry.second);
            } catch (const std::runtime_error &error) {
                throw std::runtime_error(filepath + ":" + line + ": " + error.what());
            }
        }
    } catch (const YAML::Exception &error) {
        throw std::runtime_error(filepath + ":" + std::to_string(error.mark.line + 1)
            + ": invalid YAML: " + error.msg);
    }
    if (file.bad())
        throw std::runtime_error(filepath + ": error reading configuration file");
    for (const auto *key : {"layers", "hidden_activation", "output_activation", "initialization"}) {
        if (keys.count(key) == 0)
            throw std::runtime_error(filepath + ": missing required key '" + key + "'");
    }
    return options;
}

std::string describeNetworkConfig(const NetworkConfig &config)
{
    std::ostringstream description;
    description.imbue(std::locale::classic());
    description << "layers=";
    for (std::size_t i = 0; i < config.layers.size(); ++i)
        description << (i == 0 ? "" : ",") << config.layers[i];
    description << " hidden_activation=" << config.hidden_activation
        << " output_activation=" << config.output_activation
        << " initialization=" << config.initialization
        << " learning_rate=" << config.learning_rate
        << " seed=" << config.seed
        << " optimizer=" << config.optimizer;
    return description.str();
}
