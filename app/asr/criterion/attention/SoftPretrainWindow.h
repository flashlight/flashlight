/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/app/asr/criterion/attention/WindowBase.h"

namespace fl {
namespace tasks {
namespace asr {

class SoftPretrainWindow : public WindowBase {
 public:
  explicit SoftPretrainWindow(double std);

  fl::Variable computeSingleStepWindow(
      const fl::Variable& prevAttn,
      int inputSteps,
      int batchSize,
      int step) override;

  fl::Variable computeWindowMask(int targetLen, int inputSteps, int batchSize)
      override;

 private:
  SoftPretrainWindow() = default;

  double std_;

  FL_SAVE_LOAD_WITH_BASE(WindowBase, std_)
};
} // namespace asr
} // namespace tasks
} // namespace fl

CEREAL_REGISTER_TYPE(fl::tasks::asr::SoftPretrainWindow)
