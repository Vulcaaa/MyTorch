#include "neural/CpuBackend.hpp"
#include "TestSupport.hpp"

#include <iostream>
#include <cstdlib>
#include <limits>
#include <string>

#if defined(MYTORCH_USE_BLAS) && !defined(MYTORCH_USE_ACCELERATE)
#include <cblas.h>
#endif

int main()
{
    configureCpuThreads(0);
    configureCpuThreads(1);
#ifndef MYTORCH_USE_BLAS
    test::check(std::string(cpuBackendName()) == "scalar", "incorrect scalar backend name");
    test::expectThrow<std::invalid_argument>([] { configureCpuThreads(2); });
#elif defined(MYTORCH_USE_ACCELERATE)
    configureCpuThreads(2);
    test::check(std::string(cpuBackendName()) == "accelerate", "incorrect Accelerate name");
    test::check(std::string(std::getenv("VECLIB_MAXIMUM_THREADS")) == "2",
        "Accelerate thread limit was not set before matrix operations");
#else
    configureCpuThreads(2);
    test::check(std::string(cpuBackendName()) == "openblas", "incorrect OpenBLAS name");
    test::check(openblas_get_num_threads() == 2, "OpenBLAS thread limit was not applied");
    configureCpuThreads(1);
    test::check(openblas_get_num_threads() == 1, "OpenBLAS thread limit was not reset");
#endif
    test::expectThrow<std::invalid_argument>([] {
        configureCpuThreads(static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1);
    });
    std::cout << "CPU backend tests passed\n";
}
