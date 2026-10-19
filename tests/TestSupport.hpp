#pragma once

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

namespace test {

inline void check(bool condition, const std::string &message)
{
    if (!condition)
        throw std::runtime_error(message);
}

template<typename Exception, typename Function>
std::string expectThrow(Function function)
{
    try {
        function();
    } catch (const Exception &error) {
        return error.what();
    }
    throw std::runtime_error("expected exception was not thrown");
}

class TempDirectory {
    public:
        TempDirectory()
        {
            const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
            path = std::filesystem::temp_directory_path() / ("mytorch-tests-" + std::to_string(id));
            check(std::filesystem::create_directory(path), "cannot create test directory");
        }

        ~TempDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }

        TempDirectory(const TempDirectory &) = delete;
        TempDirectory &operator=(const TempDirectory &) = delete;

        std::filesystem::path path;
};

} // namespace test
