#include "logging/Logger.hpp"
#include "TestSupport.hpp"

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <sys/resource.h>

namespace {

using std::chrono::milliseconds;
using std::chrono::system_clock;

// 2025-11-05T13:30:52Z
const system_clock::time_point reference = system_clock::from_time_t(1762349452);

void checkEntry(system_clock::time_point time, LogLevel level, const std::string& message,
    const std::string& expected)
{
    const std::string actual = formatLogEntry(time, level, message);
    test::check(actual == expected, "expected log entry '" + expected + "', got '" + actual + "'");
}

void testLevelNames()
{
    test::check(logLevelName(LogLevel::Info) == "INFO", "incorrect INFO level name");
    test::check(logLevelName(LogLevel::Warn) == "WARN", "incorrect WARN level name");
    test::check(logLevelName(LogLevel::Error) == "ERROR", "incorrect ERROR level name");
}

void testEntries()
{
    checkEntry(reference + milliseconds(123), LogLevel::Info, "Loaded model models/a.nn",
        "2025-11-05T13:30:52.123Z [INFO] Loaded model models/a.nn\n");
    checkEntry(reference, LogLevel::Warn, "slow epoch",
        "2025-11-05T13:30:52.000Z [WARN] slow epoch\n");
    checkEntry(reference + milliseconds(7), LogLevel::Error, "cannot open file",
        "2025-11-05T13:30:52.007Z [ERROR] cannot open file\n");
    checkEntry(reference + milliseconds(999), LogLevel::Info, "",
        "2025-11-05T13:30:52.999Z [INFO] \n");
    // Sub-millisecond precision is truncated, never rounded up to the next second.
    checkEntry(reference + std::chrono::microseconds(999999), LogLevel::Info, "end of second",
        "2025-11-05T13:30:52.999Z [INFO] end of second\n");
}

void testMessagesStayOnOneLine()
{
    checkEntry(reference, LogLevel::Error, "first line\nsecond line\r\nthird",
        "2025-11-05T13:30:52.000Z [ERROR] first line\\nsecond line\\r\\nthird\n");
}

std::vector<std::string> readLines(const std::filesystem::path& path)
{
    std::ifstream file(path);
    test::check(file.good(), "cannot read " + path.string());
    std::vector<std::string> lines;
    for (std::string line; std::getline(file, line);)
        lines.push_back(line);
    return lines;
}

void writeFile(const std::filesystem::path& path, const std::string& content)
{
    std::ofstream file(path);
    file << content;
    test::check(file.good(), "cannot write " + path.string());
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

bool endsWith(const std::string& text, const std::string& suffix)
{
    return text.size() >= suffix.size()
        && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void checkLine(const std::string& line, const std::string& expectedEnd)
{
    // "2025-11-05T13:30:52.123Z [INFO] message"
    const bool timestamped = line.size() > 25 && line[4] == '-' && line[7] == '-'
        && line[10] == 'T' && line[13] == ':' && line[16] == ':' && line[19] == '.'
        && line[23] == 'Z' && line[24] == ' ';
    test::check(timestamped, "log line is not timestamped: '" + line + "'");
    test::check(endsWith(line, expectedEnd),
        "log line '" + line + "' does not end with '" + expectedEnd + "'");
}

void checkSingleWarning(const std::ostringstream& diagnostics, const std::string& reason)
{
    const std::string text = diagnostics.str();
    const std::string start = "warning: logging disabled: ";
    test::check(text.rfind(start, 0) == 0 && text.find(reason) != std::string::npos
        && text.find('\n') == text.size() - 1,
        "expected one warning mentioning '" + reason + "', got '" + text + "'");
}

void testCreatesOneFilePerRun()
{
    test::TempDirectory temporary;
    const auto directory = temporary.path / "logs";
    std::ostringstream diagnostics;
    Logger logger(directory, "generation", diagnostics, reference);
    test::check(logger.enabled(), "logger is disabled");
    test::check(logger.path() == directory / "generation_2025-11-05_133052.log",
        "unexpected log file path " + logger.path().string());
    test::check(std::filesystem::is_regular_file(logger.path()), "log file was not created");
    test::check(readFile(logger.path()).empty(), "new log file is not empty");
    test::check(diagnostics.str().empty(), "unexpected diagnostics: " + diagnostics.str());
}

void testWritesEntries()
{
    test::TempDirectory temporary;
    std::ostringstream diagnostics;
    Logger logger(temporary.path, "train", diagnostics, reference);
    logger.info("Loaded model a.nn");
    logger.warn("slow epoch");
    logger.error("first\nsecond");
    logger.log(LogLevel::Info, "done");
    // Entries are flushed immediately: they are readable while the logger is open.
    const auto lines = readLines(logger.path());
    test::check(lines.size() == 4, "expected 4 log lines, got " + std::to_string(lines.size()));
    checkLine(lines[0], " [INFO] Loaded model a.nn");
    checkLine(lines[1], " [WARN] slow epoch");
    checkLine(lines[2], " [ERROR] first\\nsecond");
    checkLine(lines[3], " [INFO] done");
    test::check(logger.enabled() && diagnostics.str().empty(), "writing entries disabled the logger");
}

void testNeverOverwrites()
{
    test::TempDirectory temporary;
    const auto first = temporary.path / "train_2025-11-05_133052.log";
    writeFile(first, "earlier run\n");
    const auto target = temporary.path / "target.txt";
    writeFile(target, "keep\n");
    std::filesystem::create_symlink(target, temporary.path / "train_2025-11-05_133052_2.log");

    std::ostringstream diagnostics;
    Logger second(temporary.path, "train", diagnostics, reference);
    Logger third(temporary.path, "train", diagnostics, reference);
    second.info("second run");
    third.info("third run");
    test::check(second.path() == temporary.path / "train_2025-11-05_133052_3.log",
        "unexpected collision path " + second.path().string());
    test::check(third.path() == temporary.path / "train_2025-11-05_133052_4.log",
        "unexpected collision path " + third.path().string());
    test::check(readFile(first) == "earlier run\n", "existing log file was overwritten");
    test::check(readFile(target) == "keep\n", "log followed a symbolic link");
    test::check(readLines(second.path()).size() == 1 && readLines(third.path()).size() == 1,
        "concurrent loggers mixed their entries");
}

void testRejectsInvalidPrefixes()
{
    test::TempDirectory temporary;
    std::ostringstream diagnostics;
    for (const auto* prefix : {"", "a/b"})
        test::expectThrow<std::invalid_argument>([&] {
            Logger(temporary.path, prefix, diagnostics, reference);
        });
}

void testUnusableDirectoryDisablesLogging()
{
    test::TempDirectory temporary;
    const auto file = temporary.path / "not-a-directory";
    writeFile(file, "kept\n");
    for (const auto& directory : {file, file / "logs"}) {
        std::ostringstream diagnostics;
        Logger logger(directory, "train", diagnostics, reference);
        logger.info("ignored");
        logger.error("ignored");
        test::check(!logger.enabled() && logger.path().empty(), "unusable directory was accepted");
        checkSingleWarning(diagnostics, "not-a-directory");
    }
    test::check(readFile(file) == "kept\n", "a file was modified by a disabled logger");
}

// Limits the size of files this process may write, then restores it.
class FileSizeLimit {
    public:
        explicit FileSizeLimit(rlim_t bytes)
        {
            test::check(getrlimit(RLIMIT_FSIZE, &_original) == 0, "cannot read file size limit");
            rlimit limited = _original;
            limited.rlim_cur = bytes;
            test::check(setrlimit(RLIMIT_FSIZE, &limited) == 0, "cannot limit file size");
        }

        ~FileSizeLimit()
        {
            setrlimit(RLIMIT_FSIZE, &_original);
        }

        FileSizeLimit(const FileSizeLimit&) = delete;
        FileSizeLimit& operator=(const FileSizeLimit&) = delete;

    private:
        rlimit _original{};
};

void testWriteFailureDisablesLogging()
{
    test::TempDirectory temporary;
    std::ostringstream diagnostics;
    Logger logger(temporary.path, "train", diagnostics, reference);
    logger.info("first");
    const auto path = logger.path();
    // Without this, exceeding the limit would kill the process (SIGXFSZ) instead of
    // making the write fail.
    std::signal(SIGXFSZ, SIG_IGN);
    {
        const FileSizeLimit limit(std::filesystem::file_size(path));
        logger.info("second");
        logger.info("third");
    }
    std::signal(SIGXFSZ, SIG_DFL);
    logger.info("fourth");
    test::check(!logger.enabled(), "logger stayed enabled after a write failure");
    test::check(logger.path() == path, "log path changed after a write failure");
    checkSingleWarning(diagnostics, path.string() + ": cannot write log entry");
    const auto lines = readLines(path);
    test::check(lines.size() == 1, "expected only the first entry, got "
        + std::to_string(lines.size()) + " lines");
    checkLine(lines[0], " [INFO] first");
}

} // namespace

int main()
{
    testLevelNames();
    testEntries();
    testMessagesStayOnOneLine();
    testCreatesOneFilePerRun();
    testWritesEntries();
    testNeverOverwrites();
    testRejectsInvalidPrefixes();
    testUnusableDirectoryDisablesLogging();
    testWriteFailureDisablesLogging();
    std::cout << "Logger tests passed\n";
}
