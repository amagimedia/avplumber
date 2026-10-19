#include "OutputProgress.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
int main(int argc, char** argv) {
    assert(argc == 2);
    OutputProgress progress;
    progress.setPath(argv[1]);
    progress.setSubmitted(true);
    progress.wrote(123, 100.0);
    progress.wrote(456, 101.0);
    std::ifstream file(argv[1]);
    std::cout << file.rdbuf();
}
