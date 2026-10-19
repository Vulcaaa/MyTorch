#include "mytorch/NetworkConfig.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string_view>

namespace {

std::string_view trim(std::string_view value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

template<typename T>
T parseUnsigned(std::string_view value)
{
    T result{};
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::runtime_error("expected a non-negative integer");
    return result;
}

std::vector<std::size_t> parseLayers(std::string_view value)
{
    std::vector<std::size_t> layers;
    while (true) {
        const auto comma = value.find(',');
        const auto item = trim(value.substr(0, comma));
        if (item.empty())
            throw std::runtime_error("layer size cannot be empty");
        const auto size = parseUnsigned<std::size_t>(item);
        if (size == 0)
            throw std::runtime_error("layer sizes must be positive");
        layers.push_back(size);
        if (comma == std::string_view::npos)
            break;
        value.remove_prefix(comma + 1);
    }
    if (layers.size() < 2)
        throw std::runtime_error("layers requires at least an input and output layer");
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

void setOption(NetworkConfig &options, const std::string &key, std::string_view value)
{
    if (key == "layers") {
        options.layers = parseLayers(value);
    } else if (key == "hidden_activation") {
        if (value != "sigmoid" && value != "relu" && value != "tanh")
            throw std::runtime_error("hidden_activation must be sigmoid, relu or tanh");
        options.hidden_activation = value;
    } else if (key == "output_activation") {
        if (value != "sigmoid" && value != "softmax")
            throw std::runtime_error("output_activation must be sigmoid or softmax");
        options.output_activation = value;
    } else if (key == "learning_rate") {
        options.learning_rate = parseLearningRate(value);
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
    std::string line;
    std::size_t lineNumber = 0;
    while (std::getline(file, line)) {
        ++lineNumber;
        const auto content = trim(std::string_view(line).substr(0, line.find('#')));
        if (content.empty())
            continue;
        try {
            const auto separator = content.find('=');
            if (separator == std::string_view::npos)
                throw std::runtime_error("expected key=value");
            const std::string key(trim(content.substr(0, separator)));
            const auto value = trim(content.substr(separator + 1));
            if (key.empty() || value.empty())
                throw std::runtime_error("key and value must not be empty");
            if (!keys.insert(key).second)
                throw std::runtime_error("duplicate key '" + key + "'");
            setOption(options, key, value);
        } catch (const std::runtime_error &error) {
            throw std::runtime_error(filepath + ":" + std::to_string(lineNumber)
                + ": " + error.what());
        }
    }
    if (file.bad())
        throw std::runtime_error(filepath + ": error reading configuration file");
    for (const auto *key : {"layers", "hidden_activation", "output_activation", "initialization"}) {
        if (keys.count(key) == 0)
            throw std::runtime_error(filepath + ": missing required key '" + key + "'");
    }
    return options;
}
