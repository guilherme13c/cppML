#include "dataman.hpp"
#include "ml.hpp"
#include <ctime>
#include <iostream>

using namespace linalg;

int main(void) {
    std::srand(std::time(nullptr));

    Mat data = dload::read_csv("data/xor.csv");

    Mat datain = data.sub(0, 0, data.row_count(), data.col_count() - 1);
    Mat dataout =
        data.sub(0, data.col_count() - 1, data.row_count(), data.col_count());

    std::cout << datain;
    std::cout << dataout;

    activation act{.f = functions::sigmoidf, .f_ = derivatives::sigmoidf_};

    Layer l1 = Layer(datain.col_count(), dataout.col_count(), act);

    for (linalg::uint i = 0; i < datain.row_count(); i++) {
        linalg::Mat in(datain.col_count(), 1);

        for (linalg::uint j = 0; j < datain.col_count(); j++) {
            in.at(j, 0) = datain.at(i, j);
        }

        std::cout << "In:\t" << in;

        linalg::Mat out = l1.forward(in);
        std::cout << "Out:\t" << out;
    }

    return 0;
}
