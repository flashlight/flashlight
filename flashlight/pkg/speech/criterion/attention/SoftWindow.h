/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/attention/WindowBase.h"

namespace fl {
namespace app {
namespace asr {

class SoftWindow : public WindowBase {
 public:
  SoftWindow();
  SoftWindow(double std, double avgRate, int offset);

  Variable computeWindow(
      const Variable& prevAttn,
      int step,
      int targetLen,
      int inputSteps,
      int batchSize,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) const override;

  Variable computeVectorizedWindow(
      int targetLen,
      int inputSteps,
      int batchSize,
      const af::array& inputSizes = af::array(),
      const af::array& targetSizes = af::array()) const override;

 private:
  Variable compute(
      int targetLen,
      int inputSteps,
      int batchSize,
      const af::array& inputSizes,
      const af::array& targetSizes,
      af::array& decoderSteps) const;

  double std_;
  double avgRate_;
  int offset_;

  FL_SAVE_LOAD_WITH_BASE(WindowBase, std_, avgRate_, offset_)
};
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::SoftWindow)
