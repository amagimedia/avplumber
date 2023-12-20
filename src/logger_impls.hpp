#pragma once
#include <iostream>
#include <fstream>

#include "util.hpp"

class StdErrLogger: public Logger {
protected:
    virtual void write(const std::string &s) override {
        std::cerr << s;
    }
};

class FileLogger: public Logger {
protected:
    std::ofstream file_;
    virtual void write(const std::string &s) override {
        file_ << s;
        file_.flush();
    }
public:
    FileLogger(const std::string file_name): Logger(), file_(file_name, std::ios::binary | std::ios::app) {
    }
};
