/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace asr {

/**
 * Attention base class for encoder-decoder: decoder attends to particular
 * encoder part
 */
class AttentionBase : public fl::Container {
 public:
  AttentionBase() {}

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override {
    if (inputs.size() != 3 && inputs.size() != 4 && inputs.size() != 5) {
      throw std::invalid_argument(
          "Attention encoder-decoder: Invalid inputs size, should be 3, 4, or 5 arguments");
    }

    auto logAttnWeight = inputs.size() == 4 ? inputs[3] : Variable();
    auto xEncodedSizes = inputs.size() == 5 ? inputs[4] : Variable();
    auto res = forwardBase(
        inputs[0], inputs[1], inputs[2], logAttnWeight, xEncodedSizes);
    return {res.first, res.second};
  }

  std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn) {
    return forward(
        state,
        xEncoded,
        prevAttn,
        Variable() /* logAttnWeight */,
        Variable() /* xEncodedSizes */);
  }

  std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight) {
    return forwardBase(
        state,
        xEncoded,
        prevAttn,
        logAttnWeight,
        Variable() /* xEncodedSizes */);
  }

  virtual std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) {
    return forwardBase(state, xEncoded, prevAttn, logAttnWeight, xEncodedSizes);
  }

 protected:
  /**
   * Forward pass
   * @param state current decoder state
   * @param xEncoded encoder output = decoder input
   * @param prevAttn previous attention vector
   * @param logAttnWeight attention weights to add: finalAttn =
   * exp(logAttnWeight + logComputedAttn)
   * @param xEncodedSizes encoder output actual sizes has (1, B) dims
   * Returns <attention vector (sum = 1), summary> of sizes
   * [targetlen, seqlen, batchsize] for attention,
   * [hiddendim, targetlen, batchsize] for summary
   */
  virtual std::pair<Variable, Variable> forwardBase(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& logAttnWeight,
      const Variable& xEncodedSizes) = 0;

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};
} // namespace asr
} // namespace app
} // namespace fl

CEREAL_REGISTER_TYPE(fl::app::asr::AttentionBase)
