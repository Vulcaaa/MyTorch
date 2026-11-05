#pragma once

#include <chrono>
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
