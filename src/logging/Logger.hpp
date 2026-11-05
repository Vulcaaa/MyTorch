#pragma once

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

enum class LogLevel { Info, Warn, Error };

// Returns "INFO", "WARN" or "ERROR".
std::string_view logLevelName(LogLevel level);

// Formats one log entry as "2026-11-05T13:30:52.123Z [INFO] message\n".
// The time is written in UTC with milliseconds. Line breaks inside the message
// are escaped as \n and \r so that every entry stays on a single line.
std::string formatLogEntry(std::chrono::system_clock::time_point time, LogLevel level,
    std::string_view message);

// Writes the entries of one run to its own file.
class Logger {
    public:
        // Creates <directory>/<prefix>_<YYYY-MM-DD_HHMMSS>.log, named after the UTC
        // start time, and the directory if needed. An existing file is never
        // overwritten: a numeric suffix (_2, _3, ...) is added instead.
        Logger(const std::filesystem::path& directory, const std::string& prefix,
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now());
        ~Logger();

        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        // Appends one entry stamped with the current time and flushes it to disk.
        void log(LogLevel level, std::string_view message);
        void info(std::string_view message);
        void warn(std::string_view message);
        void error(std::string_view message);

        const std::filesystem::path& path() const;

    private:
        std::filesystem::path _path;
        std::FILE* _file = nullptr;
};
