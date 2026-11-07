#pragma once

#include <cstddef>

const char* cpuBackendName();

// Process-wide BLAS setting. Call before the first matrix operation.
// Zero preserves library/environment defaults; scalar builds accept only 0 or 1.
// A library may use fewer threads for small matrix products.
void configureCpuThreads(std::size_t threads);
