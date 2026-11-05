#include "logging/Logger.hpp"
#include "TestSupport.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

void testCreatesOneFilePerRun()
{
    test::TempDirectory temporary;
    const auto directory = temporary.path / "logs";
    Logger logger(directory, "generation", reference);
    test::check(logger.path() == directory / "generation_2025-11-05_133052.log",
        "unexpected log file path " + logger.path().string());
    test::check(std::filesystem::is_regular_file(logger.path()), "log file was not created");
    test::check(readFile(logger.path()).empty(), "new log file is not empty");
}

void testWritesEntries()
{
    test::TempDirectory temporary;
    Logger logger(temporary.path, "train", reference);
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
}

void testNeverOverwrites()
{
    test::TempDirectory temporary;
    const auto first = temporary.path / "train_2025-11-05_133052.log";
    writeFile(first, "earlier run\n");
    const auto target = temporary.path / "target.txt";
    writeFile(target, "keep\n");
    std::filesystem::create_symlink(target, temporary.path / "train_2025-11-05_133052_2.log");

    Logger second(temporary.path, "train", reference);
    Logger third(temporary.path, "train", reference);
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

void testRejectsUnusableDestinations()
{
    test::TempDirectory temporary;
    for (const auto* prefix : {"", "a/b"})
        test::expectThrow<std::invalid_argument>([&] { Logger(temporary.path, prefix, reference); });
    const auto file = temporary.path / "not-a-directory";
    writeFile(file, "");
    test::expectThrow<std::runtime_error>([&] { Logger(file, "train", reference); });
    test::expectThrow<std::runtime_error>([&] { Logger(file / "logs", "train", reference); });
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
    testRejectsUnusableDestinations();
    std::cout << "Logger tests passed\n";
}
