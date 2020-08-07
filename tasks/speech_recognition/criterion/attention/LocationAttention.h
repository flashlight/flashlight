/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/tasks/speech_recognition/criterion/attention/AttentionBase.h"

namespace fl {
namespace tasks {
namespace asr {

class SimpleLocationAttention : public AttentionBase {
 public:
  explicit SimpleLocationAttention(int convKernel);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  SimpleLocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

class LocationAttention : public AttentionBase {
 public:
  LocationAttention(int encDim, int convKernel);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  LocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

class NeuralLocationAttention : public AttentionBase {
 public:
  NeuralLocationAttention(
      int encDim,
      int attnDim,
      int convChannel,
      int convKernel);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  NeuralLocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};
} // namespace asr
} // namespace tasks
} // namespace fl

CEREAL_REGISTER_TYPE(fl::tasks::asr::SimpleLocationAttention)
CEREAL_REGISTER_TYPE(fl::tasks::asr::LocationAttention)
CEREAL_REGISTER_TYPE(fl::tasks::asr::NeuralLocationAttention)
