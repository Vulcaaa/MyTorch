#include "chess/Encoding.hpp"
#include "TestSupport.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace {

const std::string emptyBoard = "8/8/8/8/8/8/8/8";

void testEmptyBoard()
{
    std::vector<double> expected(777, 0.0);
    test::check(encodeFEN(emptyBoard + " w - - 0 1") == expected,
        "empty board with white to move must produce 777 zeros");
    expected[768] = 1.0;
    test::check(encodeFEN(emptyBoard + " b - - 0 1") == expected,
        "black to move must only activate index 768");
}

void testEveryPiece()
{
    // The first and last squares also check the direction of board traversal.
    const std::string pieces = "PNBRQKpnbrqk";
    for (std::size_t piece = 0; piece < pieces.size(); ++piece) {
        std::vector<double> expected(777, 0.0);
        expected[piece] = 1.0;
        test::check(encodeFEN(std::string(1, pieces[piece])
            + "7/8/8/8/8/8/8/8 w - - 0 1") == expected,
            "incorrect piece encoding on a8: " + std::string(1, pieces[piece]));

        expected[piece] = 0.0;
        expected[756 + piece] = 1.0;
        test::check(encodeFEN("8/8/8/8/8/8/8/7"
            + std::string(1, pieces[piece]) + " w - - 0 1") == expected,
            "incorrect piece encoding on h1: " + std::string(1, pieces[piece]));
    }
}

void testKnownPositions()
{
    std::vector<double> expected(777, 0.0);
    // a6: R, d6: k, e5: r, g5: p, f3: K, f2: P; black to move.
    for (int index : {195, 239, 345, 366, 545, 636, 768}) {
        expected[index] = 1.0;
    }
    test::check(encodeFEN("8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59") == expected,
        "dataset position has incorrect inputs");

    expected.assign(777, 0.0);
    const std::vector<int> blackBackRank = {9, 7, 8, 10, 11, 8, 7, 9};
    const std::vector<int> whiteBackRank = {3, 1, 2, 4, 5, 2, 1, 3};
    for (int file = 0; file < 8; ++file) {
        expected[file * 12 + blackBackRank[file]] = 1.0;
        expected[(8 + file) * 12 + 6] = 1.0;
        expected[(48 + file) * 12] = 1.0;
        expected[(56 + file) * 12 + whiteBackRank[file]] = 1.0;
    }
    test::check(encodeFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        == expected, "starting position has incorrect inputs");
}

void testEnPassant()
{
    for (char turn : std::string("wb")) {
        for (char file = 'a'; file <= 'h'; ++file) {
            std::vector<double> expected(777, 0.0);
            expected[768] = turn == 'b' ? 1.0 : 0.0;
            expected[769 + file - 'a'] = 1.0;
            const std::string target = std::string(1, file) + (turn == 'w' ? '6' : '3');
            test::check(encodeFEN(emptyBoard + " " + turn + " - " + target + " 0 1")
                == expected, "incorrect en passant encoding for " + target);
        }
    }
}

void testUnencodedFields()
{
    const auto expected = encodeFEN(emptyBoard + " w - - 0 1");
    // All subsets of KQkq are valid and must leave the inputs unchanged.
    const std::string rights = "KQkq";
    for (int subset = 1; subset < 16; ++subset) {
        std::string castling;
        for (int bit = 0; bit < 4; ++bit) {
            if (subset & (1 << bit)) {
                castling += rights[bit];
            }
        }
        test::check(encodeFEN(emptyBoard + " w " + castling + " - 12 59") == expected,
            "castling rights or counters changed the inputs");
    }
    test::check(encodeFEN(" \t" + emptyBoard + "\tw - - 000 001 \r\n") == expected,
        "whitespace or leading zeros rejected");
    test::check(encodeFEN(emptyBoard + " w - - 99999999999999999999 99999999999999999999")
        == expected, "large decimal counters rejected");
}

void expectInvalid(const std::string& fen, const std::string& diagnostic)
{
    const auto error = test::expectThrow<std::invalid_argument>([&] { encodeFEN(fen); });
    test::check(error.find("FEN:") == 0 && error.find(diagnostic) != std::string::npos,
        "unexpected error for " + fen + ": " + error);
}

void testInvalidFields()
{
    expectInvalid("", "6 fields");
    expectInvalid(emptyBoard + " w - - 0", "6 fields");
    expectInvalid(emptyBoard + " w - - 0 1 extra", "too many fields");

    for (const auto* board : {
        "8/8/8/8/8/8/8", "8/8/8/8/8/8/8/8/8", // rank count
        "/8/8/8/8/8/8/8", "8/8/8/8/8/8/8/", "8//8/8/8/8/8/8", // empty ranks
        "7/8/8/8/8/8/8/8", "8P/8/8/8/8/8/8/8", "P8/8/8/8/8/8/8/8",
        "44/8/8/8/8/8/8/8", "9/8/8/8/8/8/8/8", "0/8/8/8/8/8/8/8",
        "X7/8/8/8/8/8/8/8", "8/8/8/8/8/8/8/PPPPPPPPP"
    }) {
        expectInvalid(std::string(board) + " w - - 0 1", "FEN:");
    }
    for (const auto* turn : {"W", "x", "wb", "-"}) {
        expectInvalid(emptyBoard + " " + turn + " - - 0 1", "side to move");
    }
    for (const auto* castling : {"KK", "qq", "KQkqK", "K-", "--", "A"}) {
        expectInvalid(emptyBoard + " w " + castling + " - 0 1", "castling");
    }
    for (const auto* target : {"a", "a66", "i6", "A6", "d3", "d4", "--"}) {
        expectInvalid(emptyBoard + " w - " + target + " 0 1", "en passant");
    }
    expectInvalid(emptyBoard + " b - d6 0 1", "en passant");
    for (const auto* counter : {"-1", "+1", "1.5", "1x"}) {
        expectInvalid(emptyBoard + " w - - " + counter + " 1", "halfmove");
        expectInvalid(emptyBoard + " w - - 0 " + counter, "fullmove");
    }
    expectInvalid(emptyBoard + " w - - 0 0", "fullmove");
    expectInvalid(emptyBoard + " w - - 0 000", "fullmove");
}

} // namespace

int main()
{
    testEmptyBoard();
    testEveryPiece();
    testKnownPositions();
    testEnPassant();
    testUnencodedFields();
    testInvalidFields();
    std::cout << "Encoding tests passed\n";
}
