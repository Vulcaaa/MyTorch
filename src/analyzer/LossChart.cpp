#include "LossChart.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <locale>
#include <sstream>
#include <stdexcept>

#include <fcntl.h>
#include <unistd.h>

void saveLossChart(const std::string& path, const std::vector<std::array<double, 3>>& rows,
    std::size_t bestEpoch, double initialValidationLoss)
{
    if (rows.empty())
        return;
    double maximum = initialValidationLoss;
    for (const auto& row : rows)
        maximum = std::max({maximum, row[1], row[2]});
    maximum = maximum > 0 ? maximum * 1.08 : 1;
    const double last = rows.back()[0];
    const auto x = [last](double epoch) { return 85 + 720 * epoch / last; };
    const auto y = [maximum](double loss) { return 390 - 300 * loss / maximum; };
    std::ostringstream svg;
    svg.imbue(std::locale::classic());
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"920\" height=\"510\" viewBox=\"0 0 920 510\">"
        << "<rect width=\"920\" height=\"510\" fill=\"white\"/>"
        << "<g font-family=\"sans-serif\" font-size=\"13\" fill=\"#222\">"
        << "<text x=\"85\" y=\"30\" font-size=\"21\">Training and validation loss</text>"
        << "<text x=\"85\" y=\"55\">Completed training; selected checkpoint: epoch " << bestEpoch
        << "; stopped: epoch " << last << "</text>";
    const int ticks = static_cast<int>(std::min(last, 5.0));
    for (int tick = 0; tick <= ticks; ++tick) {
        const auto loss = maximum * tick / ticks;
        const auto epoch = last * tick / ticks;
        svg << "<path d=\"M85 " << y(loss) << " H805\" stroke=\"#ddd\"/>"
            << "<text x=\"75\" y=\"" << y(loss) + 4 << "\" text-anchor=\"end\">" << loss << "</text>"
            << "<text x=\"" << x(epoch) << "\" y=\"413\" text-anchor=\"middle\">" << std::round(epoch) << "</text>";
    }
    svg << "<path d=\"M85 90 V390 H805\" fill=\"none\" stroke=\"#555\"/>"
        << "<text x=\"445\" y=\"441\">Epoch</text>"
        << "<text x=\"20\" y=\"260\" transform=\"rotate(-90 20 260)\">Mean loss</text>";
    for (std::size_t column = 1; column <= 2; ++column) {
        svg << "<polyline fill=\"none\" stroke-width=\"2\" stroke=\""
            << (column == 1 ? "#1f77b4" : "#ff7f0e") << "\" points=\"";
        for (const auto& row : rows)
            svg << x(row[0]) << ',' << y(row[column]) << ' ';
        svg << "\"/>";
        if (rows.size() == 1)
            svg << "<circle cx=\"" << x(last) << "\" cy=\"" << y(rows.back()[column])
                << "\" r=\"4\" fill=\"" << (column == 1 ? "#1f77b4" : "#ff7f0e") << "\"/>";
    }
    svg << "<circle cx=\"85\" cy=\"" << y(initialValidationLoss) << "\" r=\"4\" fill=\"#ff7f0e\"/>"
        << "<path d=\"M" << x(bestEpoch) << " 90 V390\" stroke=\"#2ca02c\" stroke-dasharray=\"6 4\"/>"
        << "<path d=\"M805 90 V390\" stroke=\"#777\" stroke-dasharray=\"2 3\"/>"
        << "<text x=\"85\" y=\"475\" fill=\"#1f77b4\">Train</text>"
        << "<text x=\"155\" y=\"475\" fill=\"#c45e00\">Validation (dot: initial)</text>"
        << "<text x=\"385\" y=\"475\" fill=\"#237d23\">Dashed: selected checkpoint</text>"
        << "<text x=\"640\" y=\"475\" fill=\"#666\">Dotted: stopped</text></g></svg>";
    // Never overwrite a pre-existing chart or a path that aliases user input.
    const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (descriptor < 0)
        throw std::runtime_error("cannot create chart " + path);
    std::FILE* file = fdopen(descriptor, "w");
    if (!file) {
        close(descriptor);
        throw std::runtime_error("cannot open chart " + path);
    }
    const auto contents = svg.str();
    const bool written = std::fwrite(contents.data(), 1, contents.size(), file) == contents.size();
    const bool closed = std::fclose(file) == 0;
    if (!written || !closed)
        throw std::runtime_error("cannot write chart " + path);
}
