#ifndef NN_H_
#define NN_H_

#include <math.h>
#include <memory.h>
#include <stddef.h>
#include <stdio.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float *es;
} Mat;

// Time Complexity: O(1)
#define MAT_AT(m, i, j) ((m).es[(i) * (m).stride + (j)])

/**
 * @brief generates a random float from 0 to 1 (inclusive)
 */
float rand_float(void);
/**
 * @brief 1/(1+e^(-x))
 */
float sigmoidf(float x);
/**
 * @brief max(0, x)
 */
float reluf(float x);

Mat mat_alloc(size_t rows, size_t cols);

/**
 * @brief generates a random float from 0 to 1
 * @attention Time Complexity: O(m.rows * m.cols)
 */
void mat_rand(Mat m, float low, float high);
/**
 * @brief returns a copy of a specific row of a matrix
 * @param m Mat
 * @param row index of the row
 * @return Mat representing the row
 */
Mat mat_row(Mat m, size_t row);
/**
 * @brief Copies src to dsf
 *
 * @param src Mat
 * @param dst Mat
 * @attention O(m.rows * m.cols)
 */
void mat_copy(Mat dst, Mat src);
/**
 * @brief Fills matrix with value
 *
 * @param m Mat
 * @param v float
 * @attention O(m.rows * m.cols)
 */
void mat_fill(Mat m, const float v);
/**
 * @brief Computes the product of the matrices (dst=a*b)
 *
 * @param dst Mat to store result
 * @param a Mat
 * @param b Mat
 * @attention O(a.cols * b.rows * k), where k = a.rows = b.cols
 */
void mat_dot(Mat dst, const Mat a, const Mat b);
/**
 * @brief Computes the sum of the matrices (dst=dst+a)
 *
 * @param dst Mat to store result
 * @param a Mat
 * @attention O(a.cols * a.rows), where a.cols = dst.cols and a.rows = dst.rows
 */
void mat_sum(Mat dst, const Mat a);
/**
 * @brief Applies a function to each entry of a matrix
 *
 * @param dst Mat
 * @param func function with signature float -> float
 */
void mat_apply(Mat dst, float (*func)(float));
/**
 * @brief Prints a matrix to the stdout
 *
 * @param m Mat
 * @param name name of the matrix
 * @attention O(dst.rows * dst.cols)
 */
void mat_print(const Mat m, const char *name);
#define MAT_PRINT(m) mat_print(m, #m)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

float reluf(float x) { return fmaxf(0.0f, x); }

Mat mat_alloc(size_t rows, size_t cols) {
    NN_ASSERT(rows != 0);
    NN_ASSERT(cols != 0);

    Mat m;
    m.rows = rows;
    m.cols = cols;
    m.stride = cols;
    m.es = NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);

    return m;
}

void mat_dot(Mat dst, const Mat a, const Mat b) {
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == b.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) = 0;
            for (size_t k = 0; k < a.cols; k++) {
                MAT_AT(dst, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_sum(Mat dst, const Mat a) {
    NN_ASSERT(dst.rows == a.rows);
    NN_ASSERT(dst.cols == a.cols);

    for (size_t i = 0; i < dst.rows; i++) {
        for (size_t j = 0; j < dst.cols; j++) {
            MAT_AT(dst, i, j) += MAT_AT(a, i, j);
        }
    }
}

Mat mat_row(Mat m, size_t row) {
    return (Mat){.rows = 1,
                 .cols = m.cols,
                 .stride = m.stride,
                 .es = &MAT_AT(m, row, 0)};
}

void mat_copy(Mat dst, Mat src) {
    NN_ASSERT(dst.rows == src.rows);
    NN_ASSERT(dst.cols == src.cols);

    memcpy(dst.es, src.es, dst.rows * dst.cols * sizeof(float));
}

void mat_print(const Mat m, const char *name) {
    printf("%s = [\n", name);
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            printf("    %f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("]\n");
}

void mat_rand(Mat m, float low, float high) {
    NN_ASSERT(low <= high);

    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_fill(Mat m, const float v) {
    for (size_t i = 0; i < m.rows; i++) {
        for (size_t j = 0; j < m.cols; j++) {
            MAT_AT(m, i, j) = v;
        }
    }
}

void mat_apply(Mat dst, float (*func)(float)) {
    for (size_t i = 0; i < dst.rows * dst.cols; i++) {
        dst.es[i] = func(dst.es[i]);
    }
}

#endif // NN_IMPLEMENTATION
