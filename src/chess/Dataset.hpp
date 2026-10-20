#pragma once

#include <vector>

struct TrainingSample {
    std::vector<double> inputs; // 777 entrées
    std::vector<double> target; // 3 valeurs : Nothing, Check, Checkmate
};
