/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "flashlight/fl/nn/nn.h"

namespace fl {

/**
 * A module to implement asymmetric convolution, where only part of future or
 * part of past can be used. This is implemented with necessary pad
 * of the sequence and cut the head/tail of sequence depending on what part of
 * past/future we want to use.
 * Details see in Conv2D
 * @param futurePart_ (default 0.5): 0.5 corresponds to the symmetric
 * convolution, 0 - only past will be used, 1 - only future will be used
 * Note: currently only '0' and SAME padding are supported.
 */
class AsymmetricConv1D : public fl::Conv2D {
 public:
  AsymmetricConv1D(
      int nIn,
      int nOut,
      int wx,
      int sx = 1,
      fl::detail::IntOrPadMode px = 0,
      float futurePart = 0.5,
      int dx = 1,
      bool bias = true,
      int groups = 1);

  explicit AsymmetricConv1D(
      const fl::Variable& w,
      int sx = 1,
      fl::detail::IntOrPadMode px = 0,
      float futurePart = 0.5,
      int dx = 1,
      int groups = 1);

  AsymmetricConv1D(
      const fl::Variable& w,
      const fl::Variable& b,
      int sx = 1,
      fl::detail::IntOrPadMode px = 0,
      float futurePart = 0.5,
      int dx = 1,
      int groups = 1);

  fl::Variable forward(const fl::Variable& input) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Conv2D, futurePart_)
  float futurePart_;
  void checkParams();

  AsymmetricConv1D() = default;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AsymmetricConv1D)
