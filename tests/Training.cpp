#include "analyzer/Training.hpp"
#include "chess/Encoding.hpp"
#include "neural/ModelSerializer.hpp"
#include "TestSupport.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <csignal>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/resource.h>

namespace {

namespace fs = std::filesystem;

std::string read(const fs::path& path)
{
    std::ifstream file(path);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

NetworkConfig config(std::uint32_t seed)
{
    NetworkConfig value;
    value.layers = {777, 3};
    value.hidden_activation = "relu";
    value.output_activation = "softmax";
    value.initialization = "he";
    value.seed = seed;
    return value;
}

struct Fixture {
    test::TempDirectory directory;
    NeuralNetwork initial;
    AnalyzerArguments arguments{AnalyzerMode::Train, (directory.path / "initial.nn").string(),
        (directory.path / "train.txt").string(), (directory.path / "trained.nn").string()};
    std::vector<NeuralNetwork> updated;
    std::vector<double> validationInput = encodeFEN("r7/8/8/8/8/8/8/8 w - - 0 1");

    explicit Fixture(std::uint32_t seed = 42) : initial(config(seed))
    {
        arguments.training.validationFile = (directory.path / "validation.txt").string();
        arguments.training.historyFile = (directory.path / "history.csv").string();
        arguments.training.epochs = 8;
        arguments.training.batchSize = 2;
        arguments.training.patience = 2;
        arguments.training.minDelta = 0.1;
        std::ofstream train(arguments.chessFile);
        for (const auto* rank : {"P7", "1P6", "2P5", "3P4", "4P3"})
            train << rank << "/8/8/8/8/8/8/8 w - - 0 1 Nothing\n";
        std::ofstream(arguments.training.validationFile) << "r7/8/8/8/8/8/8/8 w - - 0 1 Nothing\n";
        ModelSerializer::save(initial, arguments.loadFile);
        auto data = nlohmann::json::parse(read(arguments.loadFile));
        data["layers"][0]["weights"] = std::vector<std::vector<double>>(3, std::vector<double>(777));
        for (int i = 0; i < 40; ++i) {
            data["layers"][0]["biases"] = std::vector<double>{i / 10.0, 0, 0};
            std::ofstream(directory.path / "fixture.nn") << data;
            updated.push_back(ModelSerializer::load((directory.path / "fixture.nn").string()));
        }
        initial = updated[0];
        ModelSerializer::save(initial, arguments.loadFile);
    }
};

// Test-only neural methods check scheduling/checkpoints, not learning.
struct Probe {
    Fixture& fixture;
    std::vector<double> validationLosses;
    std::size_t updates = 0;
    std::size_t trainEvaluations = 0;
    std::size_t validationEvaluations = 0;
    std::vector<std::vector<std::size_t>> batches;
    bool automaticSplit = false;
    std::vector<TrainingSample> automaticUpdates;
    std::vector<std::vector<TrainingSample>> automaticLossInputs;

    Probe(Fixture& value, std::vector<double> losses)
        : fixture(value), validationLosses(std::move(losses)) {}

    enum class Failure { None, InitialLoss, BatchLoss };
    Failure failure = Failure::None;

    double trainBatch(NeuralNetwork& network, const std::vector<TrainingSample>& batch)
    {
        if (automaticSplit) {
            automaticUpdates.insert(automaticUpdates.end(), batch.begin(), batch.end());
            network = fixture.updated.at(++updates);
            return 1.0;
        }
        std::vector<std::size_t> ids;
        for (const auto& sample : batch) {
            test::check(sample.inputs != fixture.validationInput, "validation was used for updates");
            test::check(sample.target == std::vector<double>({1, 0, 0}), "target conversion failed");
            ids.push_back(std::find(sample.inputs.begin(), sample.inputs.end(), 1.0) - sample.inputs.begin());
        }
        batches.push_back(std::move(ids));
        network = fixture.updated.at(++updates);
        if (failure == Failure::BatchLoss)
            return std::numeric_limits<double>::infinity();
        return 999.0; // Must not be used as the reported epoch loss.
    }

    double loss(const std::vector<TrainingSample>& samples)
    {
        if (automaticSplit) {
            automaticLossInputs.push_back(samples);
            return automaticLossInputs.size() == 1 ? 1.0 : 0.5;
        }
        if (failure == Failure::InitialLoss)
            return std::numeric_limits<double>::quiet_NaN();
        if (samples.size() == 1 && samples[0].inputs == fixture.validationInput) {
            ++validationEvaluations;
            return validationLosses.at(updates / 3);
        }
        ++trainEvaluations;
        test::check(samples.size() == 5, "epoch loss did not use the complete train set");
        return 0.25;
    }

};

Probe* activeProbe = nullptr;

TrainingResult runWithProbe(NeuralNetwork& network, const AnalyzerArguments& arguments,
    Probe& probe, std::ostream& progress)
{
    struct Selection {
        Probe* previous = activeProbe;
        explicit Selection(Probe& selected) { activeProbe = &selected; }
        ~Selection() { activeProbe = previous; }
    } selection(probe);
    Logger log(probe.fixture.directory.path / "logs", "train", progress);
    return runTraining(network, arguments, progress, log);
}

// Returns the "[LEVEL] message" part of each entry of the only log in `directory`.
std::vector<std::string> logMessages(const fs::path& directory)
{
    std::vector<fs::path> logs;
    for (const auto& entry : fs::directory_iterator(directory))
        logs.push_back(entry.path());
    test::check(logs.size() == 1, "expected one training log, got " + std::to_string(logs.size()));
    std::istringstream content(read(logs[0]));
    std::vector<std::string> messages;
    // Skip the "2025-11-05T13:30:52.123Z " timestamp, checked by the Logger tests.
    for (std::string line; std::getline(content, line);)
        messages.push_back(line.substr(25));
    return messages;
}

void checkMessages(const std::vector<std::string>& actual, const std::vector<std::string>& expected)
{
    test::check(actual.size() == expected.size(), "expected " + std::to_string(expected.size())
        + " log entries, got " + std::to_string(actual.size()));
    for (std::size_t i = 0; i < expected.size(); ++i) {
        // Epoch durations vary: "(...)" in the expectation only requires a duration in seconds.
        const auto variable = expected[i].find("(...)");
        const bool matches = variable == std::string::npos ? actual[i] == expected[i]
            : actual[i].compare(0, variable, expected[i], 0, variable) == 0
                && actual[i].size() > variable + 4 && actual[i].compare(actual[i].size() - 3, 3, " s)") == 0;
        test::check(matches, "expected log entry '" + expected[i] + "', got '" + actual[i] + "'");
    }
}

void checkSame(const NeuralNetwork& left, const NeuralNetwork& right, const Fixture& fixture)
{
    test::check(left.predict(fixture.validationInput) == right.predict(fixture.validationInput),
        "unexpected checkpoint parameters");
}

void testCheckpointAndOrdering()
{
    Fixture fixture;
    const auto source = read(fixture.arguments.loadFile);
    Probe probe(fixture, {1.0, 0.6, 0.4, 0.5, 0.45});
    auto network = fixture.initial;
    std::ostringstream progress;
    const auto result = runWithProbe(network, fixture.arguments, probe, progress);
    test::check(result.epochsCompleted == 4 && result.bestEpoch == 2 && result.earlyStopped,
        "wrong stopping epoch or best checkpoint");
    test::check(result.bestValidationLoss == 0.4, "wrong selected validation loss");
    test::check(probe.trainEvaluations == 4 && probe.validationEvaluations == 5,
        "loss evaluations must include initial validation and every completed epoch");
    test::check(probe.updates == 12, "wrong number of batch updates");
    for (std::size_t epoch = 0; epoch < 4; ++epoch) {
        test::check(probe.batches[3 * epoch].size() == 2 && probe.batches[3 * epoch + 1].size() == 2
            && probe.batches[3 * epoch + 2].size() == 1, "partial batch was not processed");
        std::vector<std::size_t> ids;
        for (std::size_t batch = 0; batch < 3; ++batch)
            ids.insert(ids.end(), probe.batches[3 * epoch + batch].begin(), probe.batches[3 * epoch + batch].end());
        std::sort(ids.begin(), ids.end());
        test::check(ids == std::vector<std::size_t>({0, 12, 24, 36, 48}), "epoch omitted or repeated a sample");
    }
    checkSame(network, fixture.updated[6], fixture);
    checkSame(ModelSerializer::load(fixture.arguments.saveFile), network, fixture);
    test::check(read(fixture.arguments.loadFile) == source, "explicit --save changed the source");
    const auto csv = read(result.historyFile);
    test::check(csv.find("epoch,train_loss,validation_loss\n1,0.25,") == 0,
        "history did not record the post-epoch train loss");
    test::check(std::count(csv.begin(), csv.end(), '\n') == 5, "wrong history length");
    test::check(progress.str().find("best_epoch=2") != std::string::npos, "missing completion summary");

    Probe repeated(fixture, {1.0, 0.6, 0.4, 0.5, 0.45});
    fixture.arguments.training.historyFile = (fixture.directory.path / "second.csv").string();
    auto second = fixture.initial;
    const auto repeat = runWithProbe(second, fixture.arguments, repeated, progress);
    test::check(probe.batches == repeated.batches && csv == read(repeat.historyFile),
        "same seed and data must reproduce sample order and history");
    checkSame(network, second, fixture);

    Fixture otherSeed(43);
    Probe changed(otherSeed, {1.0, 0.6, 0.4, 0.5, 0.45});
    auto otherNetwork = otherSeed.initial;
    runWithProbe(otherNetwork, otherSeed.arguments, changed, progress);
    test::check(probe.batches != changed.batches, "sample ordering ignored the configured seed");
}

void testStoppingAndSaveModes()
{
    Fixture fixture;
    std::ostringstream progress;
    // Small cumulative improvements reset patience, while every lower loss is checkpointed.
    fixture.arguments.training.minDelta = 0.1;
    fixture.arguments.training.epochs = 3;
    fixture.arguments.saveFile.clear();
    Probe cumulative(fixture, {1.0, 0.94, 0.88, 0.87});
    auto network = fixture.initial;
    const auto result = runWithProbe(network, fixture.arguments, cumulative, progress);
    test::check(result.epochsCompleted == 3 && !result.earlyStopped && result.bestEpoch == 3,
        "patience reference must accumulate small improvements");
    checkSame(network, fixture.updated[9], fixture);
    checkSame(ModelSerializer::load(fixture.arguments.loadFile), network, fixture);

    fixture.arguments.training.historyFile = (fixture.directory.path / "worse.csv").string();
    Probe worse(fixture, {1.0, 1.1, 1.2});
    network = fixture.initial;
    const auto noImprovement = runWithProbe(network, fixture.arguments, worse, progress);
    test::check(noImprovement.bestEpoch == 0 && noImprovement.epochsCompleted == 2,
        "initial checkpoint must be kept when no epoch improves");
    checkSame(network, fixture.initial, fixture);
    checkSame(ModelSerializer::load(fixture.arguments.loadFile), fixture.initial, fixture);
}

void testAutomaticValidation()
{
    Fixture fixture;
    fixture.arguments.training.validationFile.clear();
    fixture.arguments.training.epochs = 1;
    fixture.arguments.training.batchSize = 32;
    // Twenty groups per class, each duplicated with different unencoded FEN fields.
    std::string data;
    for (const auto* piece : {"P", "N", "B"}) {
        const auto label = std::string(piece) == "P" ? "Nothing"
            : std::string(piece) == "N" ? "Check" : "Checkmate";
        for (int square = 0; square < 20; ++square) {
            std::string board;
            for (int rank = 0; rank < 8; ++rank) {
                if (rank)
                    board += '/';
                if (rank != square / 8) {
                    board += '8';
                    continue;
                }
                const int file = square % 8;
                if (file)
                    board += std::to_string(file);
                board += piece;
                if (file < 7)
                    board += std::to_string(7 - file);
            }
            data += board + " w - - 0 1 " + label + '\n';
            data += board + " w K - 99 50 " + label + '\n';
        }
    }
    std::ofstream(fixture.arguments.chessFile) << data;
    const auto source = read(fixture.arguments.loadFile);
    auto network = fixture.initial;
    Probe probe(fixture, {});
    probe.automaticSplit = true;
    std::ostringstream progress;
    runWithProbe(network, fixture.arguments, probe, progress);
    test::check(probe.automaticLossInputs.size() == 3, "wrong automatic loss schedule");
    const auto& validation = probe.automaticLossInputs[0];
    const auto& training = probe.automaticLossInputs[1];
    test::check(training.size() == 108 && validation.size() == 12,
        "automatic split must select 90/10 groups in every class");
    std::array<std::size_t, 3> counts{};
    for (const auto& sample : validation) {
        const auto label = std::find(sample.target.begin(), sample.target.end(), 1.0) - sample.target.begin();
        ++counts[label];
        for (const auto& updated : probe.automaticUpdates)
            test::check(sample.inputs != updated.inputs, "held-out group used for updates");
    }
    test::check(counts == std::array<std::size_t, 3>{4, 4, 4}, "class stratification failed");
    test::check(probe.automaticUpdates.size() == training.size(), "automatic train coverage failed");
    for (const auto& sample : training) {
        test::check(std::count_if(probe.automaticUpdates.begin(), probe.automaticUpdates.end(),
            [&](const auto& updated) { return sample.inputs == updated.inputs; }) == 2,
            "training duplicates were omitted or repeated");
    }
    for (std::size_t i = 0; i < validation.size(); ++i)
        test::check(validation[i].inputs == probe.automaticLossInputs[2][i].inputs,
            "validation changed within an epoch");
    test::check(progress.str().find("Automatic validation split") != std::string::npos
        && progress.str().find("train=108 validation=12 seed=42") != std::string::npos,
        "automatic split counts and seed must be visible");
    test::check(read(fixture.arguments.chessFile) == data && read(fixture.arguments.loadFile) == source,
        "automatic splitting modified the source files");

    fixture.arguments.training.historyFile += ".repeat";
    Probe repeat(fixture, {});
    repeat.automaticSplit = true;
    network = fixture.initial;
    runWithProbe(network, fixture.arguments, repeat, progress);
    for (std::size_t i = 0; i < validation.size(); ++i)
        test::check(validation[i].inputs == repeat.automaticLossInputs[0][i].inputs,
            "same seed must reproduce holdout membership");

    fixture.arguments.training.historyFile += ".changed";
    network = NeuralNetwork(config(43));
    Probe changed(fixture, {});
    changed.automaticSplit = true;
    runWithProbe(network, fixture.arguments, changed, progress);
    bool different = false;
    for (std::size_t i = 0; i < validation.size(); ++i)
        different = different || validation[i].inputs != changed.automaticLossInputs[0][i].inputs;
    test::check(different, "automatic splitting ignored the model seed");

    fixture.arguments.training.historyFile += ".invalid";
    for (const auto& invalid : {
        std::string("P7/8/8/8/8/8/8/8 w - - 0 1 Nothing\nP7/8/8/8/8/8/8/8 w K - 99 50 Nothing\n"),
        data + "P7/8/8/8/8/8/8/8 w - - 0 1 Check\n"}) {
        std::ofstream(fixture.arguments.chessFile) << invalid;
        Probe rejected(fixture, {});
        rejected.automaticSplit = true;
        const auto error = test::expectThrow<std::invalid_argument>([&] {
            runWithProbe(network, fixture.arguments, rejected, progress);
        });
        test::check(error.find("automatic validation split") != std::string::npos,
            "invalid split needs a specific diagnostic");
        test::check(rejected.updates == 0 && rejected.automaticLossInputs.empty()
            && !fs::exists(fixture.arguments.training.historyFile),
            "invalid split reached training or created history");
    }
}

void testInputAndBackendFailures()
{
    Fixture fixture;
    std::ostringstream progress;
    Probe probe(fixture, {1.0, 0.5, 0.6, 0.7});
    auto network = fixture.initial;
    const auto source = read(fixture.arguments.loadFile);
    const auto fail = [&] {
        test::expectThrow<std::exception>([&] { runWithProbe(network, fixture.arguments, probe, progress); });
        test::check(probe.updates == 0, "invalid request reached a batch update");
        test::check(!fs::exists(fixture.arguments.training.historyFile), "invalid data created a history");
        checkSame(network, fixture.initial, fixture);
    };
    const auto validData = read(fixture.arguments.training.validationFile);
    // Castling and move counters differ, but the model-visible position is in train.
    std::ofstream(fixture.arguments.training.validationFile) << "P7/8/8/8/8/8/8/8 w K - 99 50 Nothing\n";
    fail();
    std::ofstream(fixture.arguments.training.validationFile) << "invalid\n";
    fail();
    std::ofstream(fixture.arguments.training.validationFile) << "r7/8/8/8/8/8/8/8 w - - 0 1\n";
    fail();
    std::ofstream(fixture.arguments.training.validationFile) << validData;
    auto missing = fixture.arguments;
    missing.training.validationFile += ".missing";
    test::expectThrow<std::runtime_error>([&] { runWithProbe(network, missing, probe, progress); });
    auto invalid = fixture.arguments;
    invalid.training.batchSize = 0;
    test::expectThrow<std::invalid_argument>([&] { runWithProbe(network, invalid, probe, progress); });

    probe.failure = Probe::Failure::InitialLoss;
    test::expectThrow<std::runtime_error>([&] { runWithProbe(network, fixture.arguments, probe, progress); });
    test::check(!fs::exists(fixture.arguments.training.historyFile), "invalid initial loss created history");
    probe.failure = Probe::Failure::BatchLoss;
    test::expectThrow<std::runtime_error>([&] { runWithProbe(network, fixture.arguments, probe, progress); });
    checkSame(network, fixture.initial, fixture);
    test::check(read(fixture.arguments.loadFile) == source && !fs::exists(fixture.arguments.saveFile),
        "failed run saved a model");
    test::check(read(fixture.arguments.training.historyFile) == "epoch,train_loss,validation_loss\n",
        "failed run did not retain a readable partial history");
}

void testHistoryAndSaveFailures()
{
    Fixture fixture;
    fixture.arguments.training.epochs = 1;
    std::ostringstream progress;
    auto network = fixture.initial;
    const auto source = read(fixture.arguments.loadFile);
    Probe probe(fixture, {1.0, 0.5});
    auto arguments = fixture.arguments;
    for (const auto& path : {arguments.loadFile, arguments.chessFile,
        arguments.training.validationFile, arguments.saveFile}) {
        arguments.training.historyFile = path;
        test::expectThrow<std::invalid_argument>([&] {
            runWithProbe(network, arguments, probe, progress);
        });
    }
    arguments = fixture.arguments;
    arguments.saveFile = arguments.chessFile;
    test::expectThrow<std::invalid_argument>([&] { runWithProbe(network, arguments, probe, progress); });
    arguments = fixture.arguments;
    arguments.training.historyFile = (fixture.directory.path / "alias.csv").string();
    fs::create_hard_link(fixture.arguments.loadFile, arguments.training.historyFile);
    test::expectThrow<std::invalid_argument>([&] { runWithProbe(network, arguments, probe, progress); });
    arguments.training.historyFile = (fixture.directory.path / "symlink.csv").string();
    fs::create_symlink("absent.csv", arguments.training.historyFile);
    test::expectThrow<std::runtime_error>([&] { runWithProbe(network, arguments, probe, progress); });
    arguments.training.historyFile = (fixture.directory.path / "alias.csv" / "history.csv").string();
    test::expectThrow<fs::filesystem_error>([&] { runWithProbe(network, arguments, probe, progress); });
    test::check(probe.updates == 0, "history creation failure reached training");
    test::check(read(fixture.arguments.loadFile) == source, "history failure overwrote source");

    arguments = fixture.arguments;
    std::ofstream(arguments.training.historyFile) << "previous run\n";
    test::expectThrow<std::runtime_error>([&] { runWithProbe(network, arguments, probe, progress); });
    test::check(read(arguments.training.historyFile) == "previous run\n", "existing history overwritten");
    arguments.training.historyFile = (fixture.directory.path / "nested" / "history.csv").string();
    arguments.saveFile = (fixture.directory.path / "missing" / "model.nn").string();
    test::expectThrow<std::runtime_error>([&] { runWithProbe(network, arguments, probe, progress); });
    checkSame(network, fixture.initial, fixture);
    test::check(read(fixture.arguments.loadFile) == source, "save failure changed source");
    test::check(read(arguments.training.historyFile).find("1,0.25,") != std::string::npos,
        "save failure lost completed history");
}

class CurrentDirectory {
    public:
        explicit CurrentDirectory(const fs::path& destination) : previous(fs::current_path())
        {
            fs::current_path(destination);
        }
        ~CurrentDirectory() { fs::current_path(previous); }
    private:
        fs::path previous;
};

void testAutomaticHistory()
{
    Fixture fixture;
    CurrentDirectory cwd(fixture.directory.path);
    fixture.arguments.training.historyFile.clear();
    fixture.arguments.training.epochs = 1;
    std::ostringstream progress;
    auto network = fixture.initial;
    Probe first(fixture, {1.0, 0.5});
    const auto result = runWithProbe(network, fixture.arguments, first, progress);
    Probe second(fixture, {1.0, 0.5});
    network = fixture.initial;
    const auto repeated = runWithProbe(network, fixture.arguments, second, progress);
    test::check(result.historyFile != repeated.historyFile, "automatic histories collided");
    test::check(fs::path(result.historyFile).parent_path().parent_path() == "logs", "wrong history root");
    test::check(read(result.historyFile) == read(repeated.historyFile), "automatic history content changed");
}

void testTrainingLog()
{
    Fixture fixture;
    Probe probe(fixture, {1.0, 0.6, 0.4, 0.5, 0.45});
    auto network = fixture.initial;
    std::ostringstream progress;
    runWithProbe(network, fixture.arguments, probe, progress);
    const auto& arguments = fixture.arguments;
    checkMessages(logMessages(fixture.directory.path / "logs"), {
        "[INFO] Train set " + arguments.chessFile + ": 5 positions",
        "[INFO] Validation set " + arguments.training.validationFile + ": 1 position",
        "[INFO] Settings: epochs=8 batch_size=2 patience=2 min_delta=0.1",
        "[INFO] Initial validation_loss=1",
        "[INFO] History: " + arguments.training.historyFile,
        "[INFO] Epoch 1/8: train_loss=0.25 validation_loss=0.6 (...)",
        "[INFO] Epoch 2/8: train_loss=0.25 validation_loss=0.4 (...)",
        "[INFO] Epoch 3/8: train_loss=0.25 validation_loss=0.5 (...)",
        "[INFO] Epoch 4/8: train_loss=0.25 validation_loss=0.45 (...)",
        "[INFO] Early stopping after epoch 4 (patience=2)",
        "[INFO] Saved model from epoch 2 (validation_loss=0.4) to " + arguments.saveFile,
    });
    test::check(progress.str().find("warning") == std::string::npos, "logging reported a failure");

    // A failed run keeps what happened before the failure, and never reports a save.
    Fixture failing;
    Probe broken(failing, {1.0});
    broken.failure = Probe::Failure::BatchLoss;
    network = failing.initial;
    test::expectThrow<std::runtime_error>([&] {
        runWithProbe(network, failing.arguments, broken, progress);
    });
    const auto messages = logMessages(failing.directory.path / "logs");
    test::check(messages.size() == 5 && messages.back() == "[INFO] History: "
        + failing.arguments.training.historyFile, "failed run logged unexpected entries");
}

void testHistoryWriteFailure()
{
    Fixture fixture;
    auto network = fixture.initial;
    Probe probe(fixture, {1.0, 0.5});
    std::ostringstream progress;
    struct rlimit original{};
    test::check(getrlimit(RLIMIT_FSIZE, &original) == 0, "cannot read file-size limit");
    auto limited = original;
    limited.rlim_cur = 0;
    const auto previousSignal = std::signal(SIGXFSZ, SIG_IGN);
    test::check(setrlimit(RLIMIT_FSIZE, &limited) == 0, "cannot set file-size limit");
    std::string diagnostic;
    try {
        diagnostic = test::expectThrow<std::runtime_error>([&] {
            runWithProbe(network, fixture.arguments, probe, progress);
        });
    } catch (...) {
        setrlimit(RLIMIT_FSIZE, &original);
        std::signal(SIGXFSZ, previousSignal);
        throw;
    }
    test::check(setrlimit(RLIMIT_FSIZE, &original) == 0, "cannot restore file-size limit");
    std::signal(SIGXFSZ, previousSignal);
    test::check(diagnostic.find("error writing training history") != std::string::npos,
        "history write failure was not reported");
    test::check(probe.updates == 0 && !fs::exists(fixture.arguments.saveFile), "history write failure saved a model");
    checkSame(network, fixture.initial, fixture);
}

} // namespace

// Linked only into Training_tests, in place of src/neural/Training.cpp.
double NeuralNetwork::trainBatch(const std::vector<TrainingSample>& batch)
{
    test::check(activeProbe != nullptr, "test neural backend not selected");
    return activeProbe->trainBatch(*this, batch);
}

double NeuralNetwork::loss(const std::vector<TrainingSample>& samples) const
{
    test::check(activeProbe != nullptr, "test neural backend not selected");
    return activeProbe->loss(samples);
}

int main()
{
    testCheckpointAndOrdering();
    testStoppingAndSaveModes();
    testAutomaticValidation();
    testInputAndBackendFailures();
    testHistoryAndSaveFailures();
    testAutomaticHistory();
    testTrainingLog();
    testHistoryWriteFailure();
    std::cout << "Training orchestration tests passed\n";
}
