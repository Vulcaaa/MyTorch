#include <iostream>
#include <string_view>

int main(int argc, char **argv)
{
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout << "USAGE\n"
                  << "    ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] "
                  << "LOADFILE CHESSFILE\n\n"
                  << "Train a neural network or classify chess positions in FEN format.\n"
                  << "Training and prediction are not implemented yet.\n";
        return 0;
    }
    std::cerr << "MyTorch: training and prediction are not implemented yet.\n"
              << "Run ./my_torch_analyzer --help for usage.\n";
    return 84;
}
