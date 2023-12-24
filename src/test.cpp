#include <iostream>
#include <ctime>
#include <linalg/linalg.hpp>

int main(int argc, char *argv[])
{
    PRNG g = PRNG(time(NULL));

    Matrix m(4, 4);
    m.randomize(g);

    std::cout << m;

    return 0;
}
