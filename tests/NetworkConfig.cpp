#include "mytorch/NetworkConfig.hpp"
#include "TestSupport.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

const std::string valid = "layers=3,2,1\nhidden_activation=relu\n"
    "output_activation=softmax\ninitialization=he\n";

void write(const std::filesystem::path &path, const std::string &content)
{
    std::ofstream file(path, std::ios::binary);
    file << content;
    test::check(file.good(), "cannot write test config");
}

} // namespace

int main()
{
    test::TempDirectory directory;
    const auto path = directory.path / "network.conf";
    write(path, valid);
    auto config = loadNetworkConfig(path.string());
    test::check(config.layers == std::vector<std::size_t>({3, 2, 1}), "incorrect layer sizes");
    test::check(config.learning_rate == 0.01 && config.seed == 42, "incorrect defaults");
    test::check(config.hidden_activation == "relu" && config.output_activation == "softmax"
        && config.initialization == "he", "incorrect options");

    write(path, "# comment\r\n\r\n layers = 3, 2, 1 # sizes\r\n"
        " hidden_activation = tanh\r\noutput_activation = sigmoid\r\n"
        "initialization = xavier\r\n learning_rate = 2.5e-2\r\nseed = 4294967295");
    config = loadNetworkConfig(path.string());
    test::check(config.layers == std::vector<std::size_t>({3, 2, 1}), "whitespace parsing failed");
    test::check(config.learning_rate == 0.025, "scientific notation parsing failed");
    test::check(config.seed == std::numeric_limits<std::uint32_t>::max(), "maximum seed rejected");
    test::check(config.hidden_activation == "tanh" && config.output_activation == "sigmoid"
        && config.initialization == "xavier", "alternative options rejected");

    for (const auto *activation : {"sigmoid", "relu", "tanh"}) {
        write(path, "layers=1,1\nhidden_activation=" + std::string(activation)
            + "\noutput_activation=sigmoid\ninitialization=xavier\nseed=0\n");
        test::check(loadNetworkConfig(path.string()).seed == 0, "zero seed rejected");
    }
    for (const auto *key : {"layers", "hidden_activation", "output_activation", "initialization"}) {
        auto missing = valid;
        const auto start = missing.find(std::string(key) + "=");
        missing.erase(start, missing.find('\n', start) - start + 1);
        write(path, missing);
        const auto error = test::expectThrow<std::runtime_error>([&] { loadNetworkConfig(path.string()); });
        test::check(error.find("missing required key '" + std::string(key) + "'") != std::string::npos,
            "missing option not identified");
    }
    const std::vector<std::string> invalidLines = {
        "unknown=1", "broken line", "=value", "seed=", "seed=1=2",
        "seed=-1", "seed=1.5", "seed=3abc", "seed=4294967296",
        "seed=9999999999999999999999999", "learning_rate=0", "learning_rate=-1",
        "learning_rate=nan", "learning_rate=inf", "learning_rate=1e999",
        "learning_rate=0.1abc", "learning_rate=text", "layers=3", "layers=3,0,1",
        "layers=3,-2,1", "layers=3,2.5,1", "layers=3,,1", "layers=3,2,",
        "layers=,3,2", "layers=3,9999999999999999999999999",
        "hidden_activation=unknown", "output_activation=relu", "initialization=unknown"
    };
    for (const auto &line : invalidLines) {
        // Put malformed options first, so duplicate-key errors cannot mask bad values.
        write(path, line + "\n" + valid);
        const auto error = test::expectThrow<std::runtime_error>([&] { loadNetworkConfig(path.string()); });
        test::check(error.find(path.string() + ":1:") != std::string::npos,
            "invalid option lacks path and line: " + line);
    }
    write(path, valid + "seed=1\nseed=2\n");
    const auto duplicate = test::expectThrow<std::runtime_error>([&] { loadNetworkConfig(path.string()); });
    test::check(duplicate.find(":6: duplicate key 'seed'") != std::string::npos,
        "duplicate key diagnostic is incorrect");
    write(path, "");
    test::expectThrow<std::runtime_error>([&] { loadNetworkConfig(path.string()); });
    test::expectThrow<std::runtime_error>([&] {
        loadNetworkConfig((directory.path / "missing.conf").string());
    });
    std::cout << "NetworkConfig tests passed\n";
}
