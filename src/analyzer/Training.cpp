#include "Training.hpp"
#include "chess/Dataset.hpp"
#include "neural/ModelSerializer.hpp"

#include <algorithm>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;

std::string runId()
{
    static std::atomic<unsigned long long> counter{0};
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    if (!gmtime_r(&time, &utc))
        throw std::runtime_error("cannot format training timestamp");
    std::ostringstream id;
    id << std::put_time(&utc, "%Y%m%dT%H%M%SZ") << '_' << getpid() << '_' << counter++;
    return id.str();
}

bool samePath(const fs::path& left, const fs::path& right)
{
    if (fs::weakly_canonical(left) == fs::weakly_canonical(right))
        return true;
    std::error_code error;
    return fs::equivalent(left, right, error) && !error;
}

const std::string& modelDestination(const AnalyzerArguments& arguments)
{
    return arguments.saveFile.empty() ? arguments.loadFile : arguments.saveFile;
}

void validateRequest(const NeuralNetwork& network, const AnalyzerArguments& arguments)
{
    if (arguments.mode != AnalyzerMode::Train)
        throw std::invalid_argument("training requires --train");
    const auto& settings = arguments.training;
    if (arguments.loadFile.empty() || arguments.chessFile.empty() || settings.validationFile.empty())
        throw std::invalid_argument("training requires model, train and validation paths");
    if (settings.epochs == 0 || settings.batchSize == 0 || settings.patience == 0
        || !std::isfinite(settings.minDelta) || settings.minDelta < 0)
        throw std::invalid_argument("invalid training settings");
    if (network.inputSize() != 777 || network.outputSize() != 3)
        throw std::invalid_argument("chess training requires 777 inputs and 3 outputs");
    if (!std::isfinite(network.config().learning_rate) || network.config().learning_rate <= 0)
        throw std::invalid_argument("learning_rate must be finite and positive");
    const auto& destination = modelDestination(arguments);
    if (destination != arguments.loadFile && samePath(destination, arguments.loadFile))
        throw std::invalid_argument("--save must not alias the source model through another path");
    for (const auto& dataset : {arguments.chessFile, settings.validationFile}) {
        if (samePath(destination, dataset) || samePath(arguments.loadFile, dataset))
            throw std::invalid_argument("model paths must not overwrite training or validation data");
    }
}

std::vector<TrainingSample> numericSamples(Dataset dataset)
{
    std::vector<TrainingSample> samples;
    samples.reserve(dataset.size());
    for (auto& sample : dataset) {
        if (!sample.target)
            throw std::invalid_argument("training and validation require labeled positions");
        samples.push_back({std::move(sample.inputs), std::move(*sample.target)});
    }
    return samples;
}

std::bitset<777> inputKey(const TrainingSample& sample)
{
    std::bitset<777> key;
    for (std::size_t i = 0; i < sample.inputs.size(); ++i)
        key[i] = sample.inputs[i] != 0;
    return key;
}

void rejectOverlap(const std::vector<TrainingSample>& training,
    const std::vector<TrainingSample>& validation)
{
    std::unordered_set<std::bitset<777>> trainInputs;
    trainInputs.reserve(training.size());
    for (const auto& sample : training)
        trainInputs.insert(inputKey(sample));
    for (const auto& sample : validation) {
        if (trainInputs.count(inputKey(sample)))
            throw std::invalid_argument("train and validation share a model-visible input");
    }
}

fs::path historyDestination(const AnalyzerArguments& arguments)
{
    if (!arguments.training.historyFile.empty())
        return arguments.training.historyFile;
    fs::create_directories("logs");
    for (int attempt = 0; attempt < 100; ++attempt) {
        const auto directory = fs::path("logs") / ("train_" + runId());
        if (fs::create_directory(directory))
            return directory / "history.csv";
    }
    throw std::runtime_error("cannot reserve a unique training history directory");
}

class HistoryFile {
    public:
        explicit HistoryFile(const AnalyzerArguments& arguments)
            : path(historyDestination(arguments))
        {
            for (const auto& protectedPath : {arguments.loadFile, arguments.chessFile,
                arguments.training.validationFile, modelDestination(arguments)}) {
                if (samePath(path, protectedPath))
                    throw std::invalid_argument("history path must not overwrite a model or dataset");
            }
            if (!path.parent_path().empty())
                fs::create_directories(path.parent_path());
            const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
            if (descriptor < 0)
                throw std::runtime_error(path.string() + ": cannot create history: " + std::strerror(errno));
            file = fdopen(descriptor, "w");
            if (!file) {
                const int error = errno;
                ::close(descriptor);
                throw std::runtime_error(path.string() + ": cannot open history stream: " + std::strerror(error));
            }
        }

        ~HistoryFile()
        {
            if (file)
                std::fclose(file);
        }

        HistoryFile(const HistoryFile&) = delete;
        HistoryFile& operator=(const HistoryFile&) = delete;

        void write(const std::string& row)
        {
            if (std::fwrite(row.data(), 1, row.size(), file) != row.size() || std::fflush(file) != 0)
                throw std::runtime_error(path.string() + ": error writing training history");
        }

        void close()
        {
            auto* closing = file;
            file = nullptr;
            if (std::fclose(closing) != 0)
                throw std::runtime_error(path.string() + ": error closing training history");
        }

        fs::path path;

    private:
        std::FILE* file = nullptr;
};

double checkedLoss(double value)
{
    if (!std::isfinite(value) || value < 0)
        throw std::runtime_error("training backend returned a nonfinite or negative loss");
    return value;
}

std::string historyRow(std::size_t epoch, double trainLoss, double validationLoss)
{
    std::ostringstream row;
    row.imbue(std::locale::classic());
    row << std::setprecision(std::numeric_limits<double>::max_digits10)
        << epoch << ',' << trainLoss << ',' << validationLoss << '\n';
    return row.str();
}

// Remove a reserved temporary model even when serialization or rename fails.
struct TemporaryModel {
    fs::path path;
    ~TemporaryModel()
    {
        std::error_code error;
        fs::remove(path, error);
    }
};

} // namespace

void saveTrainingResult(const NeuralNetwork& network, const AnalyzerArguments& arguments)
{
    if (arguments.mode != AnalyzerMode::Train)
        throw std::invalid_argument("saving is only valid in training mode");
    const auto destination = fs::weakly_canonical(modelDestination(arguments));
    for (int attempt = 0; attempt < 100; ++attempt) {
        const auto temporary = destination.parent_path()
            / ("." + destination.filename().string() + "." + runId() + ".tmp");
        const int descriptor = open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (descriptor < 0) {
            if (errno == EEXIST)
                continue;
            throw std::runtime_error(destination.string() + ": cannot create temporary model: " + std::strerror(errno));
        }
        TemporaryModel cleanup{temporary};
        if (::close(descriptor) != 0)
            throw std::runtime_error(destination.string() + ": cannot close temporary model");
        ModelSerializer::save(network, temporary.string());
        fs::rename(temporary, destination);
        return;
    }
    throw std::runtime_error("cannot reserve a temporary model file");
}

TrainingResult runTraining(NeuralNetwork& network, const AnalyzerArguments& arguments,
    std::ostream& progress)
{
    validateRequest(network, arguments);
    const auto training = numericSamples(loadDataset(arguments.chessFile, DatasetMode::Training));
    const auto validation = numericSamples(loadDataset(arguments.training.validationFile, DatasetMode::Training));
    rejectOverlap(training, validation);

    NeuralNetwork working = network;
    NeuralNetwork best = working;
    TrainingResult result;
    result.bestValidationLoss = checkedLoss(working.loss(validation));
    double patienceReference = result.bestValidationLoss;
    std::size_t staleEpochs = 0;
    std::mt19937 random(network.config().seed);
    std::vector<std::size_t> order(training.size());
    std::vector<TrainingSample> batch;
    batch.reserve(std::min(training.size(), arguments.training.batchSize));

    HistoryFile history(arguments);
    result.historyFile = history.path.string();
    progress << "Training history: " << result.historyFile << '\n';
    history.write("epoch,train_loss,validation_loss\n");
    // Use a zero-based counter to avoid overflowing the largest accepted epoch count.
    for (std::size_t index = 0; index < arguments.training.epochs; ++index) {
        const auto epoch = index + 1;
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), random);
        for (std::size_t begin = 0; begin < order.size();) {
            const auto count = std::min(arguments.training.batchSize, order.size() - begin);
            batch.clear();
            for (std::size_t i = 0; i < count; ++i)
                batch.push_back(training[order[begin + i]]);
            checkedLoss(working.trainBatch(batch));
            begin += count;
        }
        const double trainLoss = checkedLoss(working.loss(training));
        const double validationLoss = checkedLoss(working.loss(validation));
        history.write(historyRow(epoch, trainLoss, validationLoss));
        progress << "Epoch " << epoch << ": train_loss=" << trainLoss
            << " validation_loss=" << validationLoss << '\n';
        result.epochsCompleted = epoch;
        if (validationLoss < result.bestValidationLoss) {
            result.bestValidationLoss = validationLoss;
            result.bestEpoch = epoch;
            best = working;
        }
        if (validationLoss < patienceReference - arguments.training.minDelta) {
            patienceReference = validationLoss;
            staleEpochs = 0;
        } else {
            ++staleEpochs;
        }
        if (staleEpochs >= arguments.training.patience) {
            result.earlyStopped = true;
            break;
        }
    }
    history.close();
    saveTrainingResult(best, arguments);
    network = std::move(best);
    progress << "Training complete: best_epoch=" << result.bestEpoch
        << " validation_loss=" << result.bestValidationLoss
        << " saved_model=" << modelDestination(arguments) << '\n';
    return result;
}
