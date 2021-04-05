/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <arrayfire.h>

namespace fl {
namespace app {
namespace objdet {

class HungarianMatcher {
 public:
  HungarianMatcher() = default;

  HungarianMatcher(
      const float costClass,
      const float costBbox,
      const float costGiou);

  std::vector<std::pair<af::array, af::array>> compute(
      const af::array& predBoxes,
      const af::array& predLogits,
      const std::vector<af::array>& targetBoxes,
      const std::vector<af::array>& targetClasses) const;

 private:
  float costClass_;
  float costBbox_;
  float costGiou_;

  // First is SrcIdx, second is ColIdx
  std::pair<af::array, af::array> matchBatch(
      const af::array& predBoxes,
      const af::array& predLogits,
      const af::array& targetBoxes,
      const af::array& targetClasses) const;

  af::array getCostMatrix(const af::array& input, const af::array& target);
};

} // namespace objdet
} // namespace app
} // namespace fl
