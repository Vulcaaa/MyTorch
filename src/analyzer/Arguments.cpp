#include "Arguments.hpp"

#include <charconv>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string>

namespace {

bool isOption(const std::string& value)
{
    return value.rfind("--", 0) == 0;
}

std::size_t positiveInteger(const std::string& option, const std::string& value)
{
    std::size_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0)
        throw std::invalid_argument(option + " requires a positive integer");
    return result;
}

double nonnegativeNumber(const std::string& option, const std::string& value)
{
    double result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()
        || !std::isfinite(result) || result < 0)
        throw std::invalid_argument(option + " requires a finite nonnegative number");
    return result;
}

void setTrainingOption(AnalyzerArguments& arguments, const std::string& option,
    const std::string& value)
{
    if (option == "--save")
        arguments.saveFile = value;
    else if (option == "--validation")
        arguments.training.validationFile = value;
    else if (option == "--history")
        arguments.training.historyFile = value;
    else if (option == "--epochs")
        arguments.training.epochs = positiveInteger(option, value);
    else if (option == "--batch-size")
        arguments.training.batchSize = positiveInteger(option, value);
    else if (option == "--patience")
        arguments.training.patience = positiveInteger(option, value);
    else if (option == "--min-delta")
        arguments.training.minDelta = nonnegativeNumber(option, value);
}

} // namespace

AnalyzerArguments parseAnalyzerArguments(int ac, const char* const av[])
{
    if (ac < 4)
        throw std::invalid_argument("expected --predict or --train, LOADFILE and CHESSFILE");

    AnalyzerArguments arguments;
    const std::string mode = av[1];
    if (mode == "--predict")
        arguments.mode = AnalyzerMode::Predict;
    else if (mode == "--train")
        arguments.mode = AnalyzerMode::Train;
    else
        throw std::invalid_argument("mode must be --predict or --train");

    const std::set<std::string> trainingOptions = {
        "--save", "--validation", "--history", "--epochs", "--batch-size",
        "--patience", "--min-delta"
    };
    std::set<std::string> seen;
    int next = 2;
    while (next < ac && isOption(av[next])) {
        const std::string option = av[next];
        if (trainingOptions.count(option) == 0)
            throw std::invalid_argument("unknown option: " + option);
        if (arguments.mode != AnalyzerMode::Train)
            throw std::invalid_argument(option + " is only valid with --train");
        if (!seen.insert(option).second)
            throw std::invalid_argument("duplicate option: " + option);
        if (next + 1 >= ac || std::string(av[next + 1]).empty() || isOption(av[next + 1]))
            throw std::invalid_argument("missing value for " + option);
        setTrainingOption(arguments, option, av[next + 1]);
        next += 2;
    }

    if (ac != next + 2)
        throw std::invalid_argument("expected exactly LOADFILE and CHESSFILE after options");
    arguments.loadFile = av[next];
    arguments.chessFile = av[next + 1];
    for (const auto& path : {arguments.loadFile, arguments.chessFile}) {
        if (path.empty())
            throw std::invalid_argument("file paths must not be empty");
        if (isOption(path))
            throw std::invalid_argument("unexpected option: " + path);
    }
    if (arguments.mode == AnalyzerMode::Train) {
        if (arguments.saveFile.empty())
            arguments.saveFile = arguments.loadFile;
    }
    return arguments;
}
