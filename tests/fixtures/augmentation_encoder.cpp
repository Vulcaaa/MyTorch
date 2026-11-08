// Independent serialization of the production C++ encoder for augmentation QA.
#include "chess/Encoding.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main()
{
    const std::string pieces = "PNBRQKpnbrqk";
    std::string line;
    try {
        while (std::getline(std::cin, line)) {
            const auto inputs = encodeFEN(line);
            std::string key;
            for (int square = 0; square < 64; ++square) {
                char piece = '.';
                for (int channel = 0; channel < 12; ++channel)
                    if (inputs[square * 12 + channel] != 0)
                        piece = pieces[channel];
                key += piece;
            }
            key += inputs[768] == 0 ? 'w' : 'b';
            char ep = 0;
            for (int file = 0; file < 8; ++file)
                if (inputs[769 + file] != 0)
                    ep = static_cast<char>(file + 1);
            key += ep;
            std::cout.write(key.data(), key.size());
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return std::cout ? 0 : 1;
}
