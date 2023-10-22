/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/attention/AttentionBase.h"

namespace fl {
namespace pkg {
namespace speech {

class ContentAttention : public AttentionBase {
 public:
  ContentAttention(bool keyValue = false) : keyValue_(keyValue) {}

  std::unique_ptr<Module> clone() const override;

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

  std::unique_ptr<Module> clone() const override;

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
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::ContentAttention)
CEREAL_CLASS_VERSION(fl::pkg::speech::ContentAttention, 1)
CEREAL_REGISTER_TYPE(fl::pkg::speech::NeuralContentAttention)
