/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/nn/modules/Module.h"

namespace fl {

/**
 * A 2D pooling layer. This layer expects an input of shape [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C\f$, \f$N\f$]. Pooling (max or average) is performed
 * over the first and second dimensions of the input. Thus the output will be
 * of shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C\f$, \f$N\f$].
 */
class FL_API Pool2D : public UnaryModule {
 private:
  Pool2D() = default; // Intentionally private

  int xFilter_, yFilter_; // pooling dims
  int xStride_, yStride_; // stride
  int xPad_, yPad_; // padding - used iff padding mode is none
  PoolingMode mode_; // pooling type

  FL_SAVE_LOAD_WITH_BASE(
      UnaryModule,
      xFilter_,
      yFilter_,
      xStride_,
      yStride_,
      xPad_,
      yPad_,
      mode_)

 public:
  /** Construct a Pool2D layer.
   * @param wx pooling window size in the first dimension
   * @param wy pooling window size in the second dimension
   * @param sx stride in the first dimension
   * @param sy stride in the second dimension
   * @param px amount of zero-padding on both sides in the first dimension.
   * Accepts a non-negative integer value or an enum fl::PaddingMode
   * @param py amount of zero-padding on both sides in the second dimension.
   * Accepts a non-negative integer value or an enum fl::PaddingMode
   * @param mode pooling mode. Can be any of:
   * - MAX
   * - AVG_INCLUDE_PADDING
   * - AVG_EXCLUDE_PADDING
   */
  Pool2D(
      int wx,
      int wy,
      int sx = 1,
      int sy = 1,
      detail::IntOrPadMode px = 0,
      detail::IntOrPadMode py = 0,
      PoolingMode mode = PoolingMode::MAX);

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Pool2D)
