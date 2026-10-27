GENERATOR = my_torch_generator
ANALYZER = my_torch_analyzer
CXX = g++
PYTHON = python3
CPPFLAGS = -Isrc -Iinclude -MMD -MP
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror

COMMON_SRC = $(wildcard src/neural/*.cpp)
GENERATOR_SRC = $(wildcard src/generator/*.cpp)
ANALYZER_SRC = $(wildcard src/analyzer/*.cpp)
CHESS_SRC = $(wildcard src/chess/*.cpp)
CHESS_OBJ = $(CHESS_SRC:src/%.cpp=build/%.o)
COMMON_OBJ = $(COMMON_SRC:src/%.cpp=build/%.o)
GENERATOR_OBJ = $(GENERATOR_SRC:src/%.cpp=build/%.o)
ANALYZER_OBJ = $(ANALYZER_SRC:src/%.cpp=build/%.o)
TEST_SRC = $(wildcard tests/*.cpp)
TEST_OBJ = $(TEST_SRC:tests/%.cpp=build/tests/%.o)
TEST_BIN = $(TEST_SRC:tests/%.cpp=build/tests/%_tests)
DEP = $(COMMON_OBJ:.o=.d) $(GENERATOR_OBJ:.o=.d) $(ANALYZER_OBJ:.o=.d) $(TEST_OBJ:.o=.d) $(CHESS_OBJ:.o=.d)

all: $(GENERATOR) $(ANALYZER)

$(GENERATOR): $(COMMON_OBJ) $(GENERATOR_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(ANALYZER): $(COMMON_OBJ) $(ANALYZER_OBJ) $(CHESS_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/%.o: src/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

build/tests/%.o: tests/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

build/tests/%_tests: build/tests/%.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/tests/Encoding_tests: build/tests/Encoding.o build/chess/Encoding.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/tests/Dataset_tests: build/tests/Dataset.o $(CHESS_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/tests/AnalyzerArguments_tests: build/tests/AnalyzerArguments.o build/analyzer/Arguments.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/tests/AnalyzerSaving_tests: build/tests/AnalyzerSaving.o build/analyzer/Training.o $(COMMON_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

test: all $(TEST_BIN)
	@set -e; for executable in $(TEST_BIN); do ./$$executable; done
	$(PYTHON) tests/cli.py

clean:
	rm -rf build

fclean: clean
	rm -f $(GENERATOR) $(ANALYZER)

re:
	$(MAKE) fclean
	$(MAKE) all

-include $(DEP)

.PHONY: all clean fclean re test
.SECONDARY: $(TEST_OBJ)
