#include "Logger.hpp"

#include <cerrno>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr int maxFileAttempts = 1000;

std::tm utcTime(std::chrono::system_clock::time_point time)
{
    const std::time_t clock = std::chrono::system_clock::to_time_t(time);
    std::tm utc{};
    if (!gmtime_r(&clock, &utc))
        throw std::runtime_error("cannot format log timestamp");
    return utc;
}

std::string utcTimestamp(std::chrono::system_clock::time_point time)
{
    const auto seconds = std::chrono::floor<std::chrono::seconds>(time);
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(time - seconds).count();
    const std::tm utc = utcTime(seconds);
    std::ostringstream timestamp;
    timestamp << std::put_time(&utc, "%Y-%m-%dT%H:%M:%S") << '.'
        << std::setw(3) << std::setfill('0') << milliseconds << 'Z';
    return timestamp.str();
}

std::string fileTimestamp(std::chrono::system_clock::time_point time)
{
    const std::tm utc = utcTime(time);
    std::ostringstream timestamp;
    timestamp << std::put_time(&utc, "%Y-%m-%d_%H%M%S");
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

std::string formatSeconds(std::chrono::steady_clock::duration duration)
{
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    std::ostringstream text;
    text << milliseconds / 1000 << '.' << std::setw(3) << std::setfill('0') << milliseconds % 1000
        << " s";
    return text.str();
}

Logger::Logger(const std::filesystem::path& directory, const std::string& prefix,
    std::ostream& diagnostics, std::chrono::system_clock::time_point start)
    : _diagnostics(diagnostics)
{
    if (prefix.empty() || prefix.find('/') != std::string::npos)
        throw std::invalid_argument("invalid log file prefix '" + prefix + "'");
    try {
        open(directory, prefix, start);
    } catch (const std::exception& error) {
        disable(error.what());
    }
}

Logger::~Logger()
{
    if (_file)
        std::fclose(_file);
}

void Logger::open(const std::filesystem::path& directory, const std::string& prefix,
    std::chrono::system_clock::time_point start)
{
    std::filesystem::create_directories(directory);
    const std::string stem = prefix + "_" + fileTimestamp(start);
    for (int attempt = 1; attempt <= maxFileAttempts; ++attempt) {
        const std::string suffix = attempt == 1 ? "" : "_" + std::to_string(attempt);
        const auto candidate = directory / (stem + suffix + ".log");
        // O_EXCL reserves the name atomically, even against concurrent runs and symlinks.
        const int descriptor = ::open(candidate.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (descriptor < 0) {
            if (errno == EEXIST)
                continue;
            throw std::runtime_error(candidate.string() + ": cannot create log file: "
                + std::strerror(errno));
        }
        _file = fdopen(descriptor, "w");
        if (!_file) {
            const int error = errno;
            ::close(descriptor);
            throw std::runtime_error(candidate.string() + ": cannot open log file: "
                + std::strerror(error));
        }
        _path = candidate;
        return;
    }
    throw std::runtime_error("cannot reserve a unique log file in " + directory.string());
}

void Logger::disable(std::string_view reason)
{
    // Close at once so that a failed entry is never written later.
    if (_file) {
        std::fclose(_file);
        _file = nullptr;
    }
    _diagnostics << "warning: logging disabled: " << reason << '\n';
}

void Logger::log(LogLevel level, std::string_view message)
{
    if (!_file)
        return;
    try {
        const std::string entry = formatLogEntry(std::chrono::system_clock::now(), level, message);
        if (std::fwrite(entry.data(), 1, entry.size(), _file) != entry.size()
            || std::fflush(_file) != 0)
            throw std::runtime_error(_path.string() + ": cannot write log entry: "
                + std::strerror(errno));
    } catch (const std::exception& error) {
        disable(error.what());
    }
}

void Logger::info(std::string_view message)
{
    log(LogLevel::Info, message);
}

void Logger::warn(std::string_view message)
{
    log(LogLevel::Warn, message);
}

void Logger::error(std::string_view message)
{
    log(LogLevel::Error, message);
}

bool Logger::enabled() const
{
    return _file != nullptr;
}

const std::filesystem::path& Logger::path() const
{
    return _path;
}
