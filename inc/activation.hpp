#pragma once

#include <bits/stdc++.h>
#include <cmath>
#include <map>
#include <string>

typedef struct {
    float (*f)(float);
    float (*f_)(float);
} activation;

namespace functions {
float sigmoidf(float x);
float reluf(float x);
} // namespace functions

namespace derivatives {
float sigmoidf_(float x);
float reluf_(float x);
} // namespace derivatives

float functions::sigmoidf(float x) {
    return static_cast<float>(1.0f) / static_cast<float>(1.0f + expf(-x));
}

float functions::reluf(float x) {
    return static_cast<float>(std::max(x, 0.0f));
}

float derivatives::sigmoidf_(float x) {
    float s = functions::sigmoidf(x);
    return static_cast<float>(s * (1.0f - s));
}

float derivatives::reluf_(float x) {
    return static_cast<float>(x > 0 ? 1.0f : 0.0f);
}

static std::map<std::string, activation> activations = {
    {"sigmoid",
     activation{.f = functions::sigmoidf, .f_ = derivatives::sigmoidf_}},
    {"relu", activation{.f = functions::reluf, .f_ = derivatives::reluf_}}};
