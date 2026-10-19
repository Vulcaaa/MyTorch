#include <iostream>
#include <string_view>

int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "USAGE\n"
                  << "    ./my_torch_generator config_file_1 nb_1 "
                  << "[config_file_2 nb_2...]\n\n"
                  << "Create random neural networks from configuration files.\n"
                  << "Network generation is not implemented yet.\n";
        return 0;
    }
    std::cerr << "MyTorch: network generation is not implemented yet.\n"
              << "Run ./my_torch_generator --help for usage.\n";
    return 84;
}
