#include "Encoding.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

constexpr int pieceCount = 12;
constexpr int turnIndex = 64 * pieceCount;
constexpr int enPassantIndex = turnIndex + 1;
constexpr int inputCount = enPassantIndex + 8;

struct FenFields {
    std::string board;
    std::string turn;
    std::string castling;
    std::string enPassant;
    std::string halfmove;
    std::string fullmove;
};

FenFields parseFields(const std::string& fen)
{
    std::istringstream stream(fen);

    FenFields fields;

    if (!(stream >> fields.board >> fields.turn >> fields.castling
                 >> fields.enPassant >> fields.halfmove >> fields.fullmove)) {
        throw std::invalid_argument("FEN: expected 6 fields");
    }

    std::string extra;
    if (stream >> extra) {
        throw std::invalid_argument("FEN: too many fields");
    }
    return fields;
}

// Each rank goes from file a to file h. Empty squares keep their zeros.
void encodeRank(const std::string& rankText, int rank, std::vector<double>& inputs)
{
    const std::string pieces = "PNBRQKpnbrqk";
    int column = 0;
    bool previousWasDigit = false;

    for (char symbol : rankText) {
        if (symbol >= '1' && symbol <= '8') {
            if (previousWasDigit) {
                throw std::invalid_argument("FEN: consecutive empty-square counts");
            }
            column += symbol - '0';
            previousWasDigit = true;
        } else {
            const std::size_t piece = pieces.find(symbol);
            if (piece == std::string::npos) {
                throw std::invalid_argument("FEN: invalid piece symbol");
            }
            if (column >= 8) {
                throw std::invalid_argument("FEN: rank exceeds 8 squares");
            }
            const int square = rank * 8 + column;
            inputs[square * pieceCount + piece] = 1.0;
            ++column;
            previousWasDigit = false;
        }
        if (column > 8) {
            throw std::invalid_argument("FEN: rank exceeds 8 squares");
        }
    }
    if (column != 8) {
        throw std::invalid_argument("FEN: each rank must contain 8 squares");
    }
}

void encodeBoard(const std::string& board, std::vector<double>& inputs)
{
    if (std::count(board.begin(), board.end(), '/') != 7) {
        throw std::invalid_argument("FEN: expected 8 ranks");
    }

    std::istringstream stream(board);
    for (int rank = 0; rank < 8; ++rank) {
        std::string rankText;
        std::getline(stream, rankText, '/');
        encodeRank(rankText, rank, inputs);
    }
}

void encodeTurn(const std::string& turn, std::vector<double>& inputs)
{
    if (turn != "w" && turn != "b") {
        throw std::invalid_argument("FEN: side to move must be w or b");
    }
    inputs[turnIndex] = turn == "b" ? 1.0 : 0.0;
}

void validateCastling(const std::string& castling)
{
    // Castling rights are validated, but are not network inputs.
    if (castling == "-") {
        return;
    }

    const std::string rights = "KQkq";
    std::string seen;
    for (char right : castling) {
        if (rights.find(right) == std::string::npos ||
            seen.find(right) != std::string::npos) {
            throw std::invalid_argument("FEN: invalid or duplicate castling right");
        }
        seen += right;
    }
}

void encodeEnPassant(const std::string& enPassant, const std::string& turn,
                     std::vector<double>& inputs)
{
    if (enPassant == "-") {
        return;
    }

    const char expectedRank = turn == "w" ? '6' : '3';
    if (enPassant.size() != 2 ||
        enPassant[0] < 'a' || enPassant[0] > 'h' ||
        enPassant[1] != expectedRank) {
        throw std::invalid_argument("FEN: invalid en passant target for side to move");
    }
    const int column = enPassant[0] - 'a';
    inputs[enPassantIndex + column] = 1.0;
}

void validateCounters(const std::string& halfmove, const std::string& fullmove)
{
    const std::string digits = "0123456789";
    if (halfmove.find_first_not_of(digits) != std::string::npos) {
        throw std::invalid_argument("FEN: halfmove clock must be a nonnegative integer");
    }
    if (fullmove.find_first_not_of(digits) != std::string::npos ||
        fullmove.find_first_not_of('0') == std::string::npos) {
        throw std::invalid_argument("FEN: fullmove number must be a positive integer");
    }
}

} // namespace

std::vector<double> encodeFEN(const std::string& fen)
{
    const FenFields fields = parseFields(fen);
    std::vector<double> inputs(inputCount, 0.0);

    encodeBoard(fields.board, inputs);
    encodeTurn(fields.turn, inputs);
    validateCastling(fields.castling);
    encodeEnPassant(fields.enPassant, fields.turn, inputs);
    validateCounters(fields.halfmove, fields.fullmove);

    return inputs;
}
