#pragma once

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <ostream>
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

// Formats a duration in seconds with millisecond precision, such as "0.081 s".
std::string formatSeconds(std::chrono::steady_clock::duration duration);

// Writes the entries of one run to its own file.
//
// Logging never makes a command fail: if the log file cannot be created or
// written, one warning is written to `diagnostics` (stderr in the programs) and
// the logger is disabled for the rest of the run.
class Logger {
    public:
        // Creates <directory>/<prefix>_<YYYY-MM-DD_HHMMSS>.log, named after the UTC
        // start time, and the directory if needed. An existing file is never
        // overwritten: a numeric suffix (_2, _3, ...) is added instead.
        // Throws std::invalid_argument for an empty prefix or one containing '/'.
        Logger(const std::filesystem::path& directory, const std::string& prefix,
            std::ostream& diagnostics,
            std::chrono::system_clock::time_point start = std::chrono::system_clock::now());
        ~Logger();

        Logger(const Logger&) = delete;
        Logger& operator=(const Logger&) = delete;

        // Appends one entry stamped with the current time and flushes it to disk.
        // Does nothing once the logger is disabled; never throws.
        void log(LogLevel level, std::string_view message);
        void info(std::string_view message);
        void warn(std::string_view message);
        void error(std::string_view message);

        // True while entries are being written.
        bool enabled() const;
        // The log file, or an empty path if it could not be created.
        const std::filesystem::path& path() const;

    private:
        void open(const std::filesystem::path& directory, const std::string& prefix,
            std::chrono::system_clock::time_point start);
        void disable(std::string_view reason);

        std::ostream& _diagnostics;
        std::filesystem::path _path;
        std::FILE* _file = nullptr;
};
