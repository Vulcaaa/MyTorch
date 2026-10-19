GENERATOR = my_torch_generator
ANALYZER = my_torch_analyzer
CXX = g++
CPPFLAGS = -Iinclude -MMD -MP
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror

COMMON_SRC = $(wildcard src/neural/*.cpp)
GENERATOR_SRC = $(wildcard src/generator/*.cpp)
ANALYZER_SRC = $(wildcard src/analyzer/*.cpp)
COMMON_OBJ = $(COMMON_SRC:src/%.cpp=build/%.o)
GENERATOR_OBJ = $(GENERATOR_SRC:src/%.cpp=build/%.o)
ANALYZER_OBJ = $(ANALYZER_SRC:src/%.cpp=build/%.o)
DEP = $(COMMON_OBJ:.o=.d) $(GENERATOR_OBJ:.o=.d) $(ANALYZER_OBJ:.o=.d)

all: $(GENERATOR) $(ANALYZER)

$(GENERATOR): $(COMMON_OBJ) $(GENERATOR_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(ANALYZER): $(COMMON_OBJ) $(ANALYZER_OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

build/%.o: src/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build

fclean: clean
	rm -f $(GENERATOR) $(ANALYZER)

re:
	$(MAKE) fclean
	$(MAKE) all

-include $(DEP)

.PHONY: all clean fclean re
