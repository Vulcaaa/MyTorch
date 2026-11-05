#include "logging/Logger.hpp"
#include "TestSupport.hpp"

#include <chrono>
#include <iostream>
#include <string>

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

} // namespace

int main()
{
    testLevelNames();
    testEntries();
    testMessagesStayOnOneLine();
    std::cout << "Logger tests passed\n";
}
