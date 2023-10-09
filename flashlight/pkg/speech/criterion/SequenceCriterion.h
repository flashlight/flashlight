/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdexcept>

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace pkg {
namespace speech {

class SequenceCriterion : public fl::Container {
 public:
  /**
   * Find the most likely path through input using viterbi algorithm
   * https://en.wikipedia.org/wiki/Viterbi_algorithm
   */
  virtual Tensor viterbiPath(
      const Tensor& input,
      const Tensor& inputSizes = Tensor()) = 0;

  /**
   * Finds the most likely path using viterbi algorithm that is constrained
   * to go through target
   */
  virtual Tensor viterbiPathWithTarget(
      const Tensor& input,
      const Tensor& target,
      const Tensor& inputSizes = Tensor(),
      const Tensor& targetSizes = Tensor()) {
    throw std::runtime_error("Not implemented");
    return Tensor();
  }

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

using EmittingModelStatePtr = std::shared_ptr<void>;
using EmittingModelUpdateFunc = std::function<std::pair<
    std::vector<std::vector<float>>,
    std::vector<EmittingModelStatePtr>>(
    const float*,
    const int,
    const int,
    const std::vector<int>&,
    const std::vector<int>&,
    const std::vector<EmittingModelStatePtr>&,
    int&)>;

} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::SequenceCriterion)
