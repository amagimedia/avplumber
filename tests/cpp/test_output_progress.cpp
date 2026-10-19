#include "OutputProgress.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

static void show(const std::string& path) {
    std::ifstream file(path);
    std::cout << file.rdbuf();
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string path = argv[1];
    OutputProgress progress;
    progress.setPath(path);
    progress.flush();
    assert(!std::ifstream(path).good()); // Opening a destination is not publication.
    progress.wrote(100, 100.0);
    show(path);
    progress.wrote(200, 100.5);
    show(path); // Throttled, retains the prior complete receipt.
    progress.wrote(300, 101.1);
    show(path);
    progress.wrote(400, 101.4);
    progress.flush();
    show(path); // Final counters, but updated_at remains the last successful write.
    assert(!std::ifstream(path + ".tmp").good());
    OutputProgress disabled;
    disabled.wrote(1, 1.0);
    disabled.flush();
    OutputProgress unavailable;
    unavailable.setPath(path + "/missing/file.json");
    unavailable.wrote(1, 1.0); // Optional diagnostics cannot fail media output.
}
