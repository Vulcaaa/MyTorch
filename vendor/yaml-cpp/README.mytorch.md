# Bundled yaml-cpp

include/ and src/ from upstream yaml-cpp 0.8.0 (MIT license).
Source: https://github.com/jbeder/yaml-cpp/releases/tag/0.8.0
Archive: https://codeload.github.com/jbeder/yaml-cpp/tar.gz/refs/tags/0.8.0
Archive SHA-256: fbe74bbdcee21d656715688706da3c8becfd946d92cd44705cc6098bb23b3a16

The project Makefile compiles these sources into both executables. No system
yaml-cpp library, CMake, download or package installation is needed to build.

Local portability patch: src/emitterutils.cpp explicitly includes <cstdint>
for uint16_t/uint32_t on recent compilers. Vendor sources are compiled without
-Werror; project sources retain -Werror.
