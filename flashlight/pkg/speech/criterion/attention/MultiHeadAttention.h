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

class MultiHeadContentAttention : public AttentionBase {
 public:
  MultiHeadContentAttention() {}
  explicit MultiHeadContentAttention(
      int dim,
      int num_heads = 8,
      bool keyValue = false,
      bool splitInput = false);
  std::unique_ptr<Module> clone() const override;

  std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) override;

  std::string prettyString() const override;

 private:
  int numHeads_;
  bool keyValue_;
  bool splitInput_;
  FL_SAVE_LOAD_WITH_BASE(AttentionBase, numHeads_, keyValue_, splitInput_)
};
} // namespace speech
} // namespace pkg
} // namespace fl

CEREAL_REGISTER_TYPE(fl::pkg::speech::MultiHeadContentAttention)
