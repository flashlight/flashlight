/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/attention/WindowBase.h"

namespace fl {
namespace pkg {
namespace speech {

class SoftPretrainWindow : public WindowBase {
 public:
  explicit SoftPretrainWindow(double std);

  Variable computeWindow(
      const Variable& prevAttn,
      int step,
      int targetLen,
      int inputSteps,
      int batchSize,
      const Tensor& inputSizes = Tensor(),
      const Tensor& targetSizes = Tensor()) const override;

  Variable computeVectorizedWindow(
      int targetLen,
      int inputSteps,
      int batchSize,
      const Tensor& inputSizes = Tensor(),
      const Tensor& targetSizes = Tensor()) const override;

 private:
  SoftPretrainWindow() = default;

  double std_;

  Variable compute(
      int targetLen,
      int inputSteps,
      int batchSize,
      const Tensor& inputSizes,
      const Tensor& targetSizes,
      Tensor& decoderSteps) const;

  FL_SAVE_LOAD_WITH_BASE(WindowBase, std_)
};
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::SoftPretrainWindow)
