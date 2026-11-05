#include "Logger.hpp"

#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

std::string utcTimestamp(std::chrono::system_clock::time_point time)
{
    const auto seconds = std::chrono::floor<std::chrono::seconds>(time);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(time - seconds).count();
    const std::time_t clock = std::chrono::system_clock::to_time_t(seconds);
    std::tm utc{};
    if (!gmtime_r(&clock, &utc))
        throw std::runtime_error("cannot format log timestamp");
    std::ostringstream timestamp;
    timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
        << std::setw(3) << std::setfill('0') << milliseconds << 'Z';
    return timestamp.str();
}

std::string singleLine(std::string_view message)
{
    std::string line;
    line.reserve(message.size());
    for (const char character : message) {
        if (character == '\n')
            line += "\\n";
        else if (character == '\r')
            line += "\\r";
        else
            line += character;
    }
    return line;
}

} // namespace

std::string_view logLevelName(LogLevel level)
{
    switch (level) {
        case LogLevel::Info:
            return "INFO";
        case LogLevel::Warn:
            return "WARN";
        case LogLevel::Error:
            return "ERROR";
    }
    throw std::invalid_argument("unknown log level");
}

std::string formatLogEntry(std::chrono::system_clock::time_point time, LogLevel level,
    std::string_view message)
{
    std::string entry = utcTimestamp(time);
    entry += " [";
    entry += logLevelName(level);
    entry += "] ";
    entry += singleLine(message);
    entry += '\n';
    return entry;
}
