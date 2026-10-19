#pragma once

#include <random>
#include <string>

struct Layer;

namespace initialization {

enum class Method {
    Xavier,
    He
};

// Throws std::invalid_argument for an unknown name.
Method parseMethod(const std::string &name);

// Initializes weights and resets biases to zero. The caller owns the RNG.
// Throws std::invalid_argument for invalid layer dimensions or method.
void initialize(Layer &layer, Method method, std::mt19937 &rng);

} // namespace initialization
