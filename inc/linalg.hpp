#pragma once

#include <assert.h>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory.h>
#include <ostream>

namespace linalg {
typedef unsigned int uint;

#define _DEFAULT_MAT_COL_COUNT 2
#define _DEFAULT_MAT_ROW_COUNT 2

#define RANDOM_FLOAT (static_cast<float>(rand()) / static_cast<float>(RAND_MAX))

class Mat {
  private:
    float *data;

  protected:
    uint nrows, ncols;

  public:
    Mat();
    Mat(uint nrows, uint ncols);
    float &at(int row, int col);
    void fill(float v);
    void random_fill(const float low, const float high);
    Mat sub(int row0, int col0, int row1, int col1);
    Mat operator*(Mat &other);
    Mat operator*(float a);
    Mat operator+(Mat &other);
    Mat operator-(Mat &other);
    friend std::ostream &operator<<(std::ostream &os, Mat &m);
    Mat apply(std::function<float(float)> func);
    uint row_count();
    uint col_count();
};

} // namespace linalg

linalg::Mat::Mat() {
    nrows = _DEFAULT_MAT_ROW_COUNT;
    ncols = _DEFAULT_MAT_COL_COUNT;

    data = nullptr;
    data = (float *)malloc(sizeof(float) * nrows * ncols);
    assert(data != nullptr);
}

linalg::Mat::Mat(uint nrows, uint ncols) {
    this->nrows = nrows;
    this->ncols = ncols;

    data = nullptr;
    data = (float *)malloc(sizeof(float) * nrows * ncols);
    assert(data != nullptr);
}

float &linalg::Mat::at(int row, int col) {
    assert(data != nullptr);
    assert(row < (int)nrows);
    assert(col < (int)ncols);
    assert(col >= 0);
    assert(row >= 0);

    return data[row * ncols + col];
}

void linalg::Mat::fill(float v) {
    assert(ncols != 0);
    assert(nrows != 0);
    assert(data != nullptr);

    for (uint i = 0; i < nrows * ncols; i++) {
        data[i] = v;
    }
}

void linalg::Mat::random_fill(const float low, const float high) {
    assert(data != nullptr);
    assert(high > low);

    for (uint i = 0; i < nrows; i++) {
        for (uint j = 0; j < ncols; j++) {
            this->at(i, j) = RANDOM_FLOAT * (high - low) + low;
        }
    }
}

linalg::Mat linalg::Mat::sub(int row0, int col0, int row1, int col1) {
    assert(data != nullptr);
    assert(row1 <= (int)nrows);
    assert(col1 <= (int)ncols);
    assert(row1 > row0);
    assert(col1 > col0);

    linalg::Mat subMatrix(row1 - row0, col1 - col0);

    for (auto i = row0; i < row1; i++) {
        for (auto j = col0; j < col1; j++) {
            subMatrix.at(i - row0, j - col0) = this->at(i, j);
        }
    }

    return subMatrix;
}

linalg::Mat linalg::Mat::operator*(linalg::Mat &other) {
    assert(this->data != nullptr);
    assert(other.data != nullptr);
    assert(this->ncols == other.nrows);

    linalg::Mat result(this->nrows, other.ncols);
    result.fill(0);

    for (uint i = 0; i < result.nrows; i++) {
        for (uint j = 0; j < result.ncols; j++) {
            for (uint k = 0; k < this->ncols; k++) {
                result.at(i, j) += this->at(i, k) * other.at(k, j);
            }
        }
    }

    return result;
}

linalg::Mat linalg::Mat::operator*(float a) {
    assert(this->data != nullptr);

    linalg::Mat result(this->nrows, this->ncols);

    for (uint i = 0; i < this->nrows; i++) {
        for (uint j = 0; j < this->ncols; j++) {
            result.at(i, j) = this->at(i, j) * a;
        }
    }

    return result;
}

linalg::Mat linalg::Mat::operator+(linalg::Mat &other) {
    assert(this->data != nullptr);
    assert(other.data != nullptr);
    assert(this->ncols == other.ncols);
    assert(this->nrows == other.nrows);

    linalg::Mat result(this->nrows, other.ncols);

    for (uint i = 0; i < this->nrows; i++) {
        for (uint j = 0; j < this->ncols; j++) {
            result.at(i, j) = this->at(i, j) + other.at(i, j);
        }
    }

    return result;
}

linalg::Mat linalg::Mat::operator-(linalg::Mat &other) {
    assert(this->data != nullptr);
    assert(other.data != nullptr);
    assert(this->ncols == other.ncols);
    assert(this->nrows == other.nrows);

    linalg::Mat result(this->nrows, other.ncols);

    for (uint i = 0; i < this->nrows; i++) {
        for (uint j = 0; j < this->ncols; j++) {
            result.at(i, j) = this->at(i, j) - other.at(i, j);
        }
    }

    return result;
}

std::ostream &linalg::operator<<(std::ostream &os, Mat &m) {
    assert(m.data != nullptr);

    os << "[" << std::endl;

    for (uint i = 0; i < m.nrows; i++) {
        for (uint j = 0; j < m.ncols; j++) {
            os << ' ' << m.at(i, j) << ' ';
        }
        os << std::endl;
    }

    os << "]" << std::endl;

    return os;
}

linalg::Mat linalg::Mat::apply(std::function<float(float)> func) {
    assert(data != nullptr);

    linalg::Mat result(this->nrows, this->ncols);

    for (uint i = 0; i < this->nrows; i++) {
        for (uint j = 0; j < this->ncols; j++) {
            result.at(i, j) = func(this->at(i, j));
        }
    }

    return result;
}

linalg::uint linalg::Mat::row_count() { return this->nrows; }

linalg::uint linalg::Mat::col_count() { return this->ncols; }
