/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <float.h>
#include <stdint.h>
#include <limits>

#include "flashlight/fl/flashlight.h"

#include "flashlight/pkg/speech/criterion/Defines.h"

namespace fl {
namespace pkg {
namespace speech {

#define NEG_INFINITY_FLT -std::numeric_limits<float>::infinity()
#define NEG_INFINITY_DBL -std::numeric_limits<double>::infinity()

template <class T>
inline T logSumExp(T logA, T logB) {
  if (logA < logB) {
    std::swap(logA, logB);
  }
  if (logB == -std::numeric_limits<T>::infinity()) {
    return logA;
  }
  return logA + std::log1p(std::exp(logB - logA));
}

template <class T>
inline T logSumExp(T logA, T logB, T logC) {
  if (logA < logB) {
    std::swap(logA, logB);
  }
  if (logA < logC) {
    std::swap(logA, logC);
  }
  if (logB < logC) {
    std::swap(logB, logC);
  }
  if (logC == -std::numeric_limits<T>::infinity()) {
    return logSumExp(logA, logB);
  }
  return logA + std::log1p(std::exp(logB - logA) + std::exp(logC - logA));
}

template <class T>
inline void dLogSumExp(T in1, T in2, T& d1, T& d2, const float scale) {
  T maxIn = std::max(in1, in2);

  in1 = std::exp(in1 - maxIn);
  in2 = std::exp(in2 - maxIn);

  T Z = in1 + in2;

  d1 += scale * (in1 / Z);
  d2 += scale * (in2 / Z);
}

template <class T>
inline void
dLogSumExp(T in1, T in2, T in3, T& d1, T& d2, T& d3, const float scale) {
  T maxIn = std::max(std::max(in1, in2), in3);

  in1 = std::exp(in1 - maxIn);
  in2 = std::exp(in2 - maxIn);
  in3 = std::exp(in3 - maxIn);

  T Z = in1 + in2 + in3;

  d1 += scale * (in1 / Z);
  d2 += scale * (in2 / Z);
  d3 += scale * (in3 / Z);
}

int countRepeats(const int* labels, int len);

int getTargetSize(const int* labels, int len);

Tensor getTargetSizeArray(const Tensor& target, int maxSize);

lib::seq::CriterionScaleMode getCriterionScaleMode(
    const std::string& onorm,
    bool sqnorm);

// Input: N x T x B (type: float), Output: T x B (type: int)
Tensor viterbiPath(const Tensor& input, const Tensor& trans);

fl::Variable getLinearTarget(const fl::Variable& target, int T);

// apply mask to the input with proper grad.
// Mask should be the same size as input
fl::Variable applySeq2SeqMask(
    const fl::Variable& input,
    const Tensor& targetClasses,
    int padValue);
} // namespace speech
} // namespace pkg
} // namespace fl
