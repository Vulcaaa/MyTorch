GENERATOR = my_torch_generator
ANALYZER = my_torch_analyzer
CXX = g++
PYTHON = python3
CONFIG ?= configs/experiments/compare-network-sizes.yml
CPPFLAGS = -Isrc -Isrc/chess/ -Isrc/neural/ -Iinclude -MMD -MP
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -O8

# BLAS=auto uses Accelerate on macOS, OpenBLAS via pkg-config on Linux,
# and keeps the portable scalar backend when neither is available.
BLAS ?= auto
UNAME_S := $(shell uname -s)
ifeq ($(BLAS),auto)
  ifeq ($(UNAME_S),Darwin)
    BLAS_BACKEND := accelerate
  else ifeq ($(shell pkg-config --exists openblas 2>/dev/null && echo yes),yes)
    BLAS_BACKEND := openblas
  else
    BLAS_BACKEND := none
  endif
else
  BLAS_BACKEND := $(BLAS)
endif
ifeq ($(filter $(BLAS_BACKEND),none openblas accelerate),)
  $(error BLAS must be auto, none, openblas, or accelerate)
endif
ifeq ($(BLAS_BACKEND),openblas)
  BLAS_CFLAGS ?= $(shell pkg-config --cflags openblas)
  BLAS_LIBS ?= $(shell pkg-config --libs openblas)
  CPPFLAGS += -DMYTORCH_USE_BLAS $(BLAS_CFLAGS)
  LDLIBS += $(BLAS_LIBS)
endif
ifeq ($(BLAS_BACKEND),accelerate)
  CPPFLAGS += -DMYTORCH_USE_BLAS -DMYTORCH_USE_ACCELERATE -Wno-deprecated-declarations
  LDLIBS += -framework Accelerate
endif
BUILD_DIR = build/$(BLAS_BACKEND)

COMMON_SRC = $(wildcard src/neural/*.cpp)
GENERATOR_SRC = $(wildcard src/generator/*.cpp)
ANALYZER_SRC = $(wildcard src/analyzer/*.cpp)
CHESS_SRC = $(wildcard src/chess/*.cpp)
LOGGING_SRC = $(wildcard src/logging/*.cpp)
CHESS_OBJ = $(CHESS_SRC:src/%.cpp=$(BUILD_DIR)/%.o)
LOGGING_OBJ = $(LOGGING_SRC:src/%.cpp=$(BUILD_DIR)/%.o)
COMMON_OBJ = $(COMMON_SRC:src/%.cpp=$(BUILD_DIR)/%.o)
GENERATOR_OBJ = $(GENERATOR_SRC:src/%.cpp=$(BUILD_DIR)/%.o)
ANALYZER_OBJ = $(ANALYZER_SRC:src/%.cpp=$(BUILD_DIR)/%.o)
TEST_SRC = $(wildcard tests/*.cpp)
TEST_OBJ = $(TEST_SRC:tests/%.cpp=$(BUILD_DIR)/tests/%.o)
TEST_BIN = $(TEST_SRC:tests/%.cpp=$(BUILD_DIR)/tests/%_tests)
DEP = $(COMMON_OBJ:.o=.d) $(GENERATOR_OBJ:.o=.d) $(ANALYZER_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(CHESS_OBJ:.o=.d) $(LOGGING_OBJ:.o=.d)

all: $(GENERATOR) $(ANALYZER)

$(GENERATOR): $(COMMON_OBJ) $(GENERATOR_OBJ) $(LOGGING_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(ANALYZER): $(COMMON_OBJ) $(ANALYZER_OBJ) $(CHESS_OBJ) $(LOGGING_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: src/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/tests/%_tests: $(BUILD_DIR)/tests/%.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/tests/Encoding_tests: $(BUILD_DIR)/tests/Encoding.o $(BUILD_DIR)/chess/Encoding.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/tests/Dataset_tests: $(BUILD_DIR)/tests/Dataset.o $(CHESS_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/tests/AnalyzerArguments_tests: $(BUILD_DIR)/tests/AnalyzerArguments.o $(BUILD_DIR)/analyzer/Arguments.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/tests/Logger_tests: $(BUILD_DIR)/tests/Logger.o $(LOGGING_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/tests/AnalyzerSaving_tests: $(BUILD_DIR)/tests/AnalyzerSaving.o $(BUILD_DIR)/analyzer/Training.o $(COMMON_OBJ) $(CHESS_OBJ) $(LOGGING_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# This test defines scripted neural methods instead of linking src/neural/Training.cpp.
$(BUILD_DIR)/tests/Training_tests: $(BUILD_DIR)/tests/Training.o $(BUILD_DIR)/analyzer/Training.o $(filter-out $(BUILD_DIR)/neural/Training.o,$(COMMON_OBJ)) $(CHESS_OBJ) $(LOGGING_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

# The outer temporary directory also cleans artifacts after a test process crashes.
test: all $(TEST_BIN)
	@set -eu; \
	test_tmp=$$(mktemp -d); \
	trap 'rm -rf "$$test_tmp"' EXIT; \
	export TMPDIR="$$test_tmp"; \
	for executable in $(TEST_BIN); do ./$$executable; done; \
	$(PYTHON) tests/cli.py; \
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'; \
	$(PYTHON) tests/evaluation.py; \
	$(PYTHON) tests/comparison.py; \
	$(PYTHON) tests/experiment.py

# Reuse the integration test: real generation -> training -> validation evaluation.
smoke: all
	@set -eu; \
	test_tmp=$$(mktemp -d); \
	trap 'rm -rf "$$test_tmp"' EXIT; \
	export TMPDIR="$$test_tmp"; \
	$(PYTHON) tests/experiment.py ExperimentTests.test_real_builtin_training_and_before_after_evidence

experiment: all
	@test -n "$(OUTPUT)" || (echo 'Set OUTPUT=results/experiments/NAME'; exit 1)
	$(PYTHON) scripts/experiment.py "$(CONFIG)" --output "$(OUTPUT)" $(OPTIONS)

clean:
	rm -rf build

fclean: clean
	rm -f $(GENERATOR) $(ANALYZER)

re:
	$(MAKE) fclean
	$(MAKE) all

-include $(DEP)

.PHONY: all clean fclean re test smoke experiment $(GENERATOR) $(ANALYZER)
.SECONDARY: $(TEST_OBJ)
