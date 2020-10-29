/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/criterion/attention/WindowBase.h"

namespace fl {
namespace app {
namespace asr {

class SoftWindow : public WindowBase {
 public:
  SoftWindow();
  SoftWindow(double std, double avgRate, int offset);

  fl::Variable computeSingleStepWindow(
      const fl::Variable& prevAttn,
      int inputSteps,
      int batchSize,
      int step) override;

  fl::Variable computeWindowMask(int targetLen, int inputSteps, int batchSize)
      override;

 private:
  int getCenter(int step, int inputSteps);

  double std_;
  double avgRate_;
  int offset_;

  FL_SAVE_LOAD_WITH_BASE(WindowBase, std_, avgRate_, offset_)
};
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::SoftWindow)
