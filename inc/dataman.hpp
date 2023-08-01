#pragma once

#include "linalg.hpp"
#include <fstream>
#include <sstream>
#include <string>

namespace dload {
linalg::Mat read_csv(std::string path, const char sep = ',');
} // namespace dload

linalg::Mat dload::read_csv(std::string path, const char sep) {

    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + path);
    }

    std::string line;
    linalg::uint nrows = 0, ncols = 0;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string cell;
        uint colCount = 0;
        while (std::getline(iss, cell, sep)) {
            colCount++;
        }
        if (colCount > ncols) {
            ncols = colCount;
        }
        nrows++;
    }

    linalg::Mat data(nrows, ncols);

    file.clear();
    file.seekg(0, std::ios::beg);

    uint row = 0;
    while (getline(file, line)) {
        std::istringstream iss(line);
        uint col = 0;
        std::string cell;
        while (getline(iss, cell, ',')) {
            data.at(row, col) = std::stof(cell);
            col++;
        }
        row++;
    }

    file.close();

    return data;
}
