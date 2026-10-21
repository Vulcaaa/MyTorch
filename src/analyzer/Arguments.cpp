#include "Arguments.hpp"

#include <stdexcept>
#include <string>

AnalyzerArguments parseAnalyzerArguments(int argc, const char* const argv[])
{
    if (argc < 4) {
        throw std::invalid_argument("expected --predict or --train, LOADFILE and CHESSFILE");
    }

    AnalyzerArguments arguments;
    const std::string mode = argv[1];
    if (mode == "--predict") {
        arguments.mode = AnalyzerMode::Predict;
    } else if (mode == "--train") {
        arguments.mode = AnalyzerMode::Train;
    } else {
        throw std::invalid_argument("mode must be --predict or --train");
    }

    int next = 2;
    if (std::string(argv[next]) == "--save") {
        if (arguments.mode != AnalyzerMode::Train) {
            throw std::invalid_argument("--save is only valid with --train");
        }
        if (argc != 6) {
            throw std::invalid_argument("expected --train --save SAVEFILE LOADFILE CHESSFILE");
        }
        arguments.saveFile = argv[3];
        if (arguments.saveFile.empty()) {
            throw std::invalid_argument("SAVEFILE must not be empty");
        }
        next = 4;
    }

    if (argc != next + 2) {
        throw std::invalid_argument("expected exactly LOADFILE and CHESSFILE after options");
    }
    arguments.loadFile = argv[next];
    arguments.chessFile = argv[next + 1];

    for (const auto& path : {arguments.loadFile, arguments.chessFile, arguments.saveFile}) {
        if (path.rfind("--", 0) == 0) {
            throw std::invalid_argument("unexpected option: " + path);
        }
    }
    if (arguments.loadFile.empty() || arguments.chessFile.empty()) {
        throw std::invalid_argument("file paths must not be empty");
    }
    if (arguments.mode == AnalyzerMode::Train && arguments.saveFile.empty()) {
        arguments.saveFile = arguments.loadFile;
    }
    return arguments;
}
