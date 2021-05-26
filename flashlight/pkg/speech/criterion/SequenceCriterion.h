/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
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
  virtual af::array viterbiPath(
      const af::array& input,
      const af::array& inputSizes = af::array()) = 0;

  /**
   * Finds the most likely path using viterbi algorithm that is constrained
   * to go through target
   */
  virtual af::array viterbiPathWithTarget(
      const af::array& input,
      const af::array& target,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) {
    throw std::runtime_error("Not implemented");
    return af::array();
  }

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

typedef std::shared_ptr<void> AMStatePtr;
typedef std::function<
    std::pair<std::vector<std::vector<float>>, std::vector<AMStatePtr>>(
        const float*,
        const int,
        const int,
        const std::vector<int>&,
        const std::vector<AMStatePtr>&,
        int&)>
    AMUpdateFunc;
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::SequenceCriterion)
