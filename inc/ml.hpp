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
               linalg::uint epochs);
    void backwards(const linalg::Mat &loss, float learning_rate,
                   linalg::Mat inputdata);
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

    for (linalg::uint i = 0; i < layers.size(); i++) {
        result = layers[i].forward(result);
    }

    return result.transpose();
}

// TODO: DEBUG & TEST
void NeuralNetwork::backwards(const linalg::Mat &loss, float learning_rate,
                              linalg::Mat inputdata) {
    linalg::Mat gradient = loss;

    for (int i = layers.size() - 1; i >= 0; i--) {
        linalg::Mat input =
            (i == 0) ? inputdata.transpose() : this->layers[i - 1].input;
        std::cout << gradient;
        std::cout << input;

        linalg::Mat delta_weights = gradient * input.transpose();
        std::cout << 1;
        linalg::Mat delta_biases = gradient;

        // Update weights and biases using gradients and learning rate
        layers[i].w = layers[i].w - delta_weights * learning_rate;
        std::cout << 2;
        layers[i].b = layers[i].b - delta_biases * learning_rate;
        std::cout << 3;

        // Compute gradient for the previous layer
        gradient = layers[i].w.transpose() * gradient;
        std::cout << 4;
        std::cout << '\n';
    }
}

void NeuralNetwork::train(linalg::Mat &inputdata, linalg::Mat &outputdata,
                          float learning_rate, linalg::uint epochs) {
    assert(inputdata.row_count() == outputdata.row_count());

    for (linalg::uint epoch = 0; epoch < epochs; epoch++) {
        for (linalg::uint sample = 0; sample < inputdata.row_count();
             sample++) {
            linalg::Mat input =
                inputdata.sub(sample, 0, sample + 1, inputdata.col_count());
            linalg::Mat target =
                outputdata.sub(sample, 0, sample + 1, outputdata.col_count())
                    .transpose();

            linalg::Mat prediction = this->predict(input);

            linalg::Mat loss = prediction - target;

            std::cout << "backwards:\n";
            this->backwards(loss, learning_rate, inputdata);
        }
    }
}
