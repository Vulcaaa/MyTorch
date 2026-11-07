#include "CpuBackend.hpp"

#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#if defined(MYTORCH_USE_BLAS) && !defined(MYTORCH_USE_ACCELERATE)
#include <cblas.h>
#endif

const char* cpuBackendName()
{
#ifndef MYTORCH_USE_BLAS
    return "scalar";
#elif defined(MYTORCH_USE_ACCELERATE)
    return "accelerate";
#else
    return "openblas";
#endif
}

void configureCpuThreads(std::size_t threads)
{
    if (threads > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("--threads exceeds the supported integer range");
    if (threads == 0)
        return;
#ifndef MYTORCH_USE_BLAS
    if (threads != 1)
        throw std::invalid_argument("--threads > 1 requires a BLAS build (BLAS=openblas or BLAS=accelerate)");
#elif defined(MYTORCH_USE_ACCELERATE)
    // Set before the first Accelerate operation; vecLib caches its environment.
    if (setenv("VECLIB_MAXIMUM_THREADS", std::to_string(threads).c_str(), 1) != 0)
        throw std::runtime_error("cannot configure Accelerate thread limit");
#else
    openblas_set_num_threads(static_cast<int>(threads));
#endif
}
