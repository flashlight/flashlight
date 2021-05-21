/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/attention/AttentionBase.h"

namespace fl {
namespace app {
namespace asr {

class ContentAttention : public AttentionBase {
 public:
  ContentAttention(bool keyValue = false) : keyValue_(keyValue) {}

  std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) override;

  std::string prettyString() const override;

 private:
  bool keyValue_;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase, fl::versioned(keyValue_, 1))
};

class NeuralContentAttention : public AttentionBase {
 public:
  NeuralContentAttention() {}
  explicit NeuralContentAttention(int dim, int layers = 1);

  std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::ContentAttention)
CEREAL_CLASS_VERSION(fl::app::asr::ContentAttention, 1)
CEREAL_REGISTER_TYPE(fl::app::asr::NeuralContentAttention)
