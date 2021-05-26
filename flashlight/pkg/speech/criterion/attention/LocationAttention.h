/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/speech/criterion/attention/AttentionBase.h"

namespace fl {
namespace pkg {
namespace speech {

class SimpleLocationAttention : public AttentionBase {
 public:
  explicit SimpleLocationAttention(int convKernel);

  std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) override;

  std::string prettyString() const override;

 private:
  SimpleLocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

class LocationAttention : public AttentionBase {
 public:
  LocationAttention(int encDim, int convKernel);

  std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) override;

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

  std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) override;

  std::string prettyString() const override;

 private:
  NeuralLocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::SimpleLocationAttention)
CEREAL_REGISTER_TYPE(fl::pkg::speech::LocationAttention)
CEREAL_REGISTER_TYPE(fl::pkg::speech::NeuralLocationAttention)
