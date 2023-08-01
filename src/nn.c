#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

float td[] = {
    0, 0, 0,
    0, 1, 1, 
    1, 0, 1, 
    1, 1, 0,
};

int main(void) {
    srand(time(0));

    size_t stride = 3;
    size_t n = sizeof(td) / sizeof(td[0]) / stride;
    Mat ti = {.rows = n, .cols = 2, .stride = stride, .es = td};
    Mat to = {.rows = n, .cols = 1, .stride = stride, .es = td + 2};

    MAT_PRINT(ti);
    MAT_PRINT(to);

    Mat x = mat_alloc(1, 2);
    Mat w1 = mat_alloc(2, 2);
    Mat b1 = mat_alloc(1, 2);
    Mat a1 = mat_alloc(1, 2);
    Mat w2 = mat_alloc(2, 1);
    Mat b2 = mat_alloc(1, 1);
    Mat a2 = mat_alloc(1, 1);

    mat_rand(w1, 0, 1);
    mat_rand(b1, 0, 1);
    mat_rand(w2, 0, 1);
    mat_rand(b2, 0, 1);

    MAT_AT(x, 0, 0) = 0;
    MAT_AT(x, 0, 1) = 1;

    mat_dot(a1, x, w1);
    mat_sum(a1, b1);
    mat_apply(a1, sigmoidf);
    mat_dot(a2, a1, w2);
    mat_sum(a2, b2);
    mat_apply(a2, sigmoidf);

    MAT_PRINT(w1);
    MAT_PRINT(b1);
    MAT_PRINT(a1);
    MAT_PRINT(w2);
    MAT_PRINT(b2);
    MAT_PRINT(a2);

    Mat c = mat_alloc(1, w1.cols);
    c = mat_row(w1, 1);
    MAT_PRINT(c);

    return 0;
}
