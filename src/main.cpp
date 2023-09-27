#include "dataman.hpp"
#include "ml.hpp"
#include <ctime>
#include <iostream>

using namespace linalg;

int main(void) {
    std::srand(std::time(nullptr));

    Mat data = dload::read_csv("data/xor.csv");

    Mat datain = data.sub(0, 0, data.row_count(), data.col_count() - 1);
    Mat dataout = data.get_col(2);

    std::cout << datain;
    std::cout << dataout;

    std::vector<Layer> layers = {
        Layer(datain.col_count(), 4, activations["relu"]),
        Layer(4, 1, activations["relu"])};

    NeuralNetwork model =
        NeuralNetwork(datain.col_count(), dataout.col_count(), layers);

    linalg::Mat test = datain.get_row(0);
    test = model.predict(test);

    std::cout << test;

    return 0;
}
