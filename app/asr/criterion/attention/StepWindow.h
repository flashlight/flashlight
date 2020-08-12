/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "WindowBase.h"

namespace fl {
namespace tasks {
namespace asr {

class StepWindow : public WindowBase {
 public:
  StepWindow();
  StepWindow(int sMin, int sMax, double vMin, double vMax);

  fl::Variable computeSingleStepWindow(
      const fl::Variable& prevAttn,
      int inputSteps,
      int batchSize,
      int step) override;

  fl::Variable computeWindowMask(int targetLen, int inputSteps, int batchSize)
      override;

 private:
  int sMin_;
  int sMax_;
  double vMin_;
  double vMax_;

  FL_SAVE_LOAD_WITH_BASE(WindowBase, sMin_, sMax_, vMin_, vMax_)
};
} // namespace asr
} // namespace tasks
} // namespace fl

CEREAL_REGISTER_TYPE(fl::tasks::asr::StepWindow)
