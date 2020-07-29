/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "runtime/Helpers.h"

#include <random>

#include "common/Utils.h"

namespace w2l {

template <class T>
std::vector<std::string> afMatrixToStrings(const af::array& arr, T terminator) {
  int L = arr.dims(0); // padded length of string
  int N = arr.dims(1); // number of strings
  std::vector<std::string> result;
  auto values = w2l::afToVector<T>(arr);
  for (int i = 0; i < N; ++i) {
    const T* row = &values[i * L];
    int len = 0;
    while (len < L && row[len] != terminator) {
      ++len;
    }
    result.emplace_back(row, row + len);
  }
  return result;
}

std::unordered_set<int64_t>
getTrainEvalIds(int64_t dsSize, double pctTrainEval, int64_t seed) {
  std::mt19937_64 rng(seed);
  std::bernoulli_distribution dist(pctTrainEval / 100.0);
  std::unordered_set<int64_t> result;
  for (int64_t i = 0; i < dsSize; ++i) {
    if (dist(rng)) {
      result.insert(i);
    }
  }
  return result;
}

std::vector<std::string> readSampleIds(const af::array& arr) {
  return afMatrixToStrings<int>(arr, -1);
}

} // namespace w2l
