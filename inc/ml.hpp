#pragma once

#include "activation.hpp"
#include "linalg.hpp"
#include <vector>

class Layer {
  public:
    linalg::uint input_size;
    linalg::uint output_size;
    linalg::Mat input;
    activation activationFunction;
    linalg::Mat w;
    linalg::Mat b;
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
    this->activationFunction = f;
}

linalg::Mat Layer::forward(linalg::Mat &input) {
    assert(input.col_count() == 1);
    assert(input.row_count() == this->input_size);

    return (this->w * input + this->b).apply(this->activationFunction.f);
}

class NeuralNetwork {
  private:
    linalg::uint inputSize;
    linalg::uint outputSize;
    std::vector<Layer> layers;

  public:
    NeuralNetwork(linalg::uint inputSize, linalg::uint outputSize,
                  std::vector<Layer> &layers);
    linalg::Mat predict(linalg::Mat &input);
    void train(linalg::Mat &datain, linalg::Mat &dataout, float learning_rate,
               linalg::uint epochs,
               linalg::Mat (*loss)(const linalg::Mat &predicted,
                                   const linalg::Mat &actual));
};

NeuralNetwork::NeuralNetwork(linalg::uint inputSize, linalg::uint outputSize,
                             std::vector<Layer> &layers) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->layers = layers;
}

linalg::Mat NeuralNetwork::predict(linalg::Mat &input) {
    assert(input.row_count() == 1);
    assert(input.col_count() == inputSize);

    linalg::Mat result = input.transpose();

    for (auto i = 0; i < layers.size(); i++) {
        result = layers[i].forward(result);
    }

    return result.transpose();
}
