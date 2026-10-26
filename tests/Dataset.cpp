#include "chess/Dataset.hpp"
#include "chess/Encoding.hpp"
#include "TestSupport.hpp"

#include <fstream>
#include <iostream>

namespace {

const std::string white = "8/8/8/8/8/8/8/8 w - - 0 1";
const std::string black = "8/8/8/8/8/8/8/8 b - - 0 1";

void writeFile(const std::string& path, const std::string& contents)
{
    std::ofstream file(path);
    file << contents;
    test::check(static_cast<bool>(file), "cannot write dataset fixture");
}

void testFormatsAndOrder(const std::string& path)
{
    writeFile(path, " \t\r\n" + white + "\r\n\n" + black + " Check White\n"
        + white + " Checkmate Black");
    const auto samples = loadDataset(path);
    test::check(samples.size() == 3, "blank lines must be skipped");
    test::check(samples[0].inputs == encodeFEN(white) && !samples[0].target,
        "unlabeled position must keep an absent target");
    test::check(samples[1].inputs == encodeFEN(black)
        && samples[1].target == std::vector<double>({0, 1, 0}),
        "position order or Check target is incorrect");
    test::check(samples[2].inputs == encodeFEN(white)
        && samples[2].target == std::vector<double>({0, 0, 1}),
        "position order or Checkmate target is incorrect");
}

void testLabels(const std::string& path)
{
    const std::vector<std::string> labels = {
        "Nothing", "Check", "Check White", "Check Black",
        "Checkmate", "Checkmate White", "Checkmate Black"
    };
    for (std::size_t i = 0; i < labels.size(); ++i) {
        writeFile(path, white + " " + labels[i] + " \t\r\n");
        std::vector<double> expected(3, 0.0);
        expected[i == 0 ? 0 : (i < 4 ? 1 : 2)] = 1.0;
        for (auto mode : {DatasetMode::Training, DatasetMode::Prediction}) {
            const auto samples = loadDataset(path, mode);
            test::check(samples[0].target == expected, "incorrect target for " + labels[i]);
        }
    }
}

void testErrors(const std::string& path)
{
    const std::vector<std::string> invalidLines = {
        "8/8/8/8/8/8/8/8 w - - 0", // Missing FEN field.
        "8/8/8/8/8/8/8 w - - 0 1 Nothing", // Malformed board.
        white + " Unknown", white + " Nothing White",
        white + " Check Blue", white + " Check White extra",
        white + " Check Checkmate", white + " Nothing extra",
        white + " check", white + " Checkmate Black extra"
    };
    for (const auto& line : invalidLines) {
        writeFile(path, "\n" + line + "\n");
        for (auto mode : {DatasetMode::Training, DatasetMode::Prediction}) {
            const auto error = test::expectThrow<std::runtime_error>([&] { loadDataset(path, mode); });
            test::check(error.find(path + ":2:") != std::string::npos,
                "invalid data must identify its path and line");
        }
    }

    writeFile(path, white + " Nothing\n\n" + black);
    const auto error = test::expectThrow<std::runtime_error>([&] {
        loadDataset(path, DatasetMode::Training);
    });
    test::check(error.find(path + ":3:") != std::string::npos
        && error.find("missing training label") != std::string::npos,
        "training must reject a missing label with its location");

    for (const auto* contents : {"", "\n \t\r\n"}) {
        writeFile(path, contents);
        const auto emptyError = test::expectThrow<std::runtime_error>([&] { loadDataset(path); });
        test::check(emptyError.find(path) != std::string::npos
            && emptyError.find("no positions") != std::string::npos,
            "empty datasets must be rejected explicitly");
    }
    const auto missing = path + ".missing";
    const auto missingError = test::expectThrow<std::runtime_error>([&] { loadDataset(missing); });
    test::check(missingError.find(missing) != std::string::npos,
        "missing-file errors must identify the path");
}

void testReadFailure(const std::string& directory)
{
    // Reading a directory fails even when opening it as a stream succeeds.
    const auto error = test::expectThrow<std::runtime_error>([&] { loadDataset(directory); });
    test::check(error.find(directory) != std::string::npos
        && (error.find("cannot read dataset") != std::string::npos
            || error.find("cannot open dataset") != std::string::npos),
        "unreadable input must report an I/O error, not an empty dataset");
}

} // namespace

int main()
{
    test::TempDirectory temporary;
    const auto path = (temporary.path / "positions.txt").string();
    testFormatsAndOrder(path);
    testLabels(path);
    testErrors(path);
    testReadFailure(temporary.path.string());
    std::cout << "Dataset tests passed\n";
}
