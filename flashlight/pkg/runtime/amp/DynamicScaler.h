/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>
#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace pkg {
namespace runtime {

/**
 * Dynamically scales up the training loss as well as all the gradients in back
 * propagation to avoid underflow in AMP training. The gradients are then scaled
 * down in the optimization step. A typical use case would be like:
 *
 * DynamicScaler dynamicScaler;
 * for (sample : dataset) {
 *   opt.zerograd();
 *
 *   output = model(sample);
 *   loss = criterion(output);
 *   if (!dynamicScaler.scale(loss)) {
 *     continue;
 *   }
 *
 *   loss.backward();
 *   if (!dynamicScaler.unscale(model.params())) {
 *       continue;
 *   }
 *   opt.step();
 * }
 */
class DynamicScaler {
 public:
  DynamicScaler(
      double initFactor,
      double maxFactor,
      unsigned int updateInterval);

  /*
   * Scale loss before back propagation.
   */
  fl::Variable scale(const fl::Variable& loss);

  /*
   * Unscale the gradients after back propagation.
   * Return false when NAN or INF occurs in gradients and halve the scale
   * factor, true otherwise.
   */
  bool unscale(std::vector<fl::Variable>& params);

  /*
   * Increase scale factor
   */
  void update();

  /*
   * Return the current scale factor
   */
  double getScaleFactor() const;

 private:
  double scaleFactor_;
  // The maximum value of scaleFactor_.
  double maxScaleFactor_;
  // Number of iterations without changing scaleFactor_.
  unsigned int successCounter_{0};
  // Double up the scaleFactor_ when successCounter_ equals updateInterval_.
  unsigned int updateInterval_;

  FL_SAVE_LOAD(scaleFactor_, maxScaleFactor_, updateInterval_, successCounter_)
  DynamicScaler() = default;
};

} // namespace runtime
} // namespace pkg
} // namespace fl
