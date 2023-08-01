#pragma once

#include "activation.hpp"
#include "linalg.hpp"

class Layer {
  private:
    linalg::uint input_size;
    linalg::uint output_size;
    linalg::Mat w;
    linalg::Mat b;
    activation f;

  public:
    Layer(linalg::uint input_size, linalg::uint output_size, activation &f);
    linalg::Mat forward(linalg::Mat &input);
};

Layer::Layer(linalg::uint input_size, linalg::uint output_size, activation &f) {
    assert(input_size > 0);
    assert(output_size > 0);

    this->w = linalg::Mat(output_size, input_size);
    this->w.fill(1);

    this->b = linalg::Mat(output_size, 1);
    this->b.fill(1);

    this->input_size = input_size;
    this->output_size = output_size;
    this->f = f;
}

linalg::Mat Layer::forward(linalg::Mat &input) {
    assert(input.col_count() == 1);
    assert(input.row_count() == this->input_size);

    return (this->w * input + this->b).apply(this->f.f);
}
