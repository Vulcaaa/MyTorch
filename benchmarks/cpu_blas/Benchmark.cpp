#include "chess/Dataset.hpp"
#include "neural/ModelSerializer.hpp"
#include "neural/NeuralNetwork.hpp"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

namespace {

using Clock = std::chrono::steady_clock;

double seconds(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::vector<TrainingSample> load(const std::string& path)
{
    std::vector<TrainingSample> result;
    for (auto& sample : loadDataset(path, DatasetMode::Training))
        result.push_back({std::move(sample.inputs), std::move(*sample.target)});
    return result;
}

double accuracy(const NeuralNetwork& network, const std::vector<TrainingSample>& samples)
{
    std::size_t correct = 0;
    for (const auto& sample : samples) {
        const auto p = network.predict(sample.inputs);
        correct += std::max_element(p.begin(), p.end()) - p.begin()
            == std::max_element(sample.target.begin(), sample.target.end()) - sample.target.begin();
    }
    return static_cast<double>(correct) / samples.size();
}

} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc != 7)
            throw std::invalid_argument("expected train, validation, epochs, batch size, seed, model output");
        const auto train = load(argv[1]);
        const auto validation = load(argv[2]);
        const auto epochs = std::stoull(argv[3]);
        const auto batchSize = std::stoull(argv[4]);
        if (epochs == 0 || batchSize == 0 || train.empty() || validation.empty())
            throw std::invalid_argument("datasets, epochs and batch size must be nonempty/positive");
        NetworkConfig config;
        config.layers = {777, 128, 64, 3};
        config.hidden_activation = "sigmoid";
        config.output_activation = "softmax";
        config.learning_rate = 0.01;
        config.initialization = "xavier";
        config.seed = static_cast<unsigned int>(std::stoul(argv[5]));
        NeuralNetwork network(config);
        // Untimed warm-up on a copy: no change to initialization or shuffle state.
        auto warmup = network;
        std::vector<std::size_t> warmOrder(std::min<std::size_t>(batchSize, train.size()));
        std::iota(warmOrder.begin(), warmOrder.end(), 0);
        warmup.trainBatch(train, warmOrder, 0, warmOrder.size());
        warmup.loss(validation);

        std::mt19937 random(config.seed);
        std::vector<std::size_t> order(train.size());
        nlohmann::json history = nlohmann::json::array();
        double trainLoss = 0.0, validationLoss = 0.0;
        const auto totalStart = Clock::now();
        for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
            const auto epochStart = Clock::now();
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), random);
            const auto updateStart = Clock::now();
            for (std::size_t begin = 0; begin < train.size();) {
                const auto count = std::min<std::size_t>(batchSize, train.size() - begin);
                network.trainBatch(train, order, begin, count);
                begin += count;
            }
            const double updateSeconds = seconds(updateStart);
            const auto lossStart = Clock::now();
            trainLoss = network.loss(train);
            validationLoss = network.loss(validation);
            const double lossSeconds = seconds(lossStart);
            history.push_back({{"epoch", epoch + 1}, {"train_loss", trainLoss},
                {"validation_loss", validationLoss}, {"update_seconds", updateSeconds},
                {"loss_seconds", lossSeconds}, {"epoch_seconds", seconds(epochStart)}});
        }
        const double totalSeconds = seconds(totalStart);
        ModelSerializer::save(network, argv[6]);
        nlohmann::json result{{"total_seconds", totalSeconds}, {"history", history},
            {"train_loss", trainLoss}, {"validation_loss", validationLoss},
            {"validation_accuracy", accuracy(network, validation)}};
        std::cout << result.dump(2) << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 84;
    }
}
