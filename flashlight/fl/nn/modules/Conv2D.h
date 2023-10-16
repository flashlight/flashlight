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

namespace detail {
class ConvBenchmarks;
}

/**
 * Applies a 2D convolution over an 4D input along its first two dimensions.
 * This layer expects an input of shape [\f$X_{in}\f$, \f$Y_{in}\f$,
 * \f$C_{in}\f$, \f$N\f$] where \f$C_{in}\f$ is the number of input channels,
 * and generates an output of shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$]
 * where \f$C_{out}\f$ is the number of output channels,
 * \f[X_{out} = \frac{X_{in} + 2 \times X_{pad} - (1 + (X_{filter} - 1) \times
 * X_{dilation})}{X_{stride}} + 1\f]
 * \f[Y_{out} = \frac{Y_{in} + 2 \times Y_{pad} - (1 + (Y_{filter} - 1) \times
 * Y_{dilation})}{Y_{stride}} + 1\f]
 *
 * Two modes for zero-padding are supported:
 *
 * - AF_PADDING_NONE: no padding
 * - AF_PADDING_SAME: \f$X_{pad}\f$ and \f$Y_{pad}\f$ are
 * dynamically chosen so that
 * \f[X_{out} = \lceil{\frac{X_{in}}{X_{stride}}}\rceil,
 *  Y_{out} = \lceil{\frac{Y_{in}}{Y_{stride}}}\rceil\f]
 */
class FL_API Conv2D : public UnaryModule {
 private:
  FL_SAVE_LOAD_WITH_BASE(
      UnaryModule,
      nIn_,
      nOut_,
      xFilter_,
      yFilter_,
      xStride_,
      yStride_,
      xPad_,
      yPad_,
      fl::versioned(xDilation_, 1),
      fl::versioned(yDilation_, 1),
      bias_,
      groups_)

  void initialize();

 protected:
  Conv2D() = default;
  int nIn_, nOut_; // in/op channels
  int xFilter_, yFilter_; // filter dims
  int xStride_, yStride_; // stride
  int xPad_, yPad_; // padding
  int xDilation_{1}, yDilation_{1}; // dilation
  bool bias_;
  int groups_;

 public:
  /**
   * Constructs a Conv2D module
   *
   * @param n_in \f$C_{in}\f$, the number of channels in the input
   * @param n_out \f$C_{out}\f$, the number of channels in the output
   * @param wx the size of the first dimension of the convolving kernel
   * @param wy the size of the second dimension of the convolving kernel
   * @param sx the stride of the convolution along the first dimension
   * @param sy the stride of the convolution along the second dimension
   * @param px the amount of zero-padding added to the both sides of the first
   *  dimension of the input. Accepts a non-negative integer value or an enum
   * fl::PaddingMode
   * @param py the amount of zero-padding added to the both sides of the second
   *  dimension of the input. Accepts a non-negative integer value or an enum
   * fl::PaddingMode
   * @param dx dilation of the convolution along the first kernel dimension. A
   *  dilation of 1 is equivalent to a standard convolution along this axis.
   * @param dy dilation of the convolution along the second kernel dimension. A
   *  dilation of 1 is equivalent to a standard convolution along this axis.
   * @param bias a boolean value that controls whether to add a learnable bias
   *  to the output
   * @param groups the number of groups that the input and output channels
   *  are divided into for restricting the connectivity between input and output
   *  channels. If `groups` > 1, the the output channels in the i-th group will
   *  be only connected to the input channels in the i-th group
   */
  Conv2D(
      int n_in,
      int n_out,
      int wx,
      int wy,
      int sx = 1,
      int sy = 1,
      detail::IntOrPadMode px = 0,
      detail::IntOrPadMode py = 0,
      int dx = 1,
      int dy = 1,
      bool bias = true,
      int groups = 1);

  /**
   * Constructs a Conv2D module with a kernel `Variable` tensor. No bias term
   * will be applied to the output.
   *
   * @param w the kernel `Variable` tensor. The shape should be
   *  [\f$kerneldim_0\f$, \f$kerneldim_1\f$, \f$C_{in}\f$, \f$C_{out}\f$].
   * @param sx the stride of the convolution along the first dimension
   * @param sy the stride of the convolution along the second dimension
   * @param px the amount of zero-padding added to the both sides of the first
   *  dimension of the input. Accepts a non-negative integer value or an enum
   * fl::PaddingMode
   * @param py the amount of zero-padding added to the both sides of the second
   *  dimension of the input. Accepts a non-negative integer value or an enum
   * fl::PaddingMode
   * @param dx dilation of the convolution along the first kernel dimension. A
   *  dilation of 1 is equivalent to a standard convolution along this axis.
   * @param dy dilation of the convolution along the second kernel dimension. A
   *  dilation of 1 is equivalent to a standard convolution along this axis.
   * @param groups the number of groups that the input and output channels
   *  are divided into for restricting the connectivity between input and output
   *  channels. If `groups` > 1, the the output channels in the i-th group will
   *  be only connected to the input channels in the i-th group.
   */
  explicit Conv2D(
      const Variable& w,
      int sx = 1,
      int sy = 1,
      detail::IntOrPadMode px = 0,
      detail::IntOrPadMode py = 0,
      int dx = 1,
      int dy = 1,
      int groups = 1);

  /**
   * Constructs a Conv2D module with a kernel `Variable` tensor and a bias
   * `Variable` tensor.
   *
   * @param w the kernel `Variable` tensor. The shape should be
   *  [\f$kerneldim_0\f$, \f$kerneldim_1\f$, \f$C_{in}\f$, \f$C_{out}\f$].
   * @param b the bias `Variable` tensor. The shape should be
   *  [\f$1\f$, \f$1\f$, \f$C_{out}\f$, \f$1\f$].
   * @param sx the stride of the convolution along the first dimension
   * @param sy the stride of the convolution along the second dimension
   * @param px the amount of zero-padding added to the both sides of the first
   *  dimension of the input. Accepts a non-negative integer value or an enum
   * fl::PaddingMode
   * @param py the amount of zero-padding added to the both sides of the second
   *  dimension of the input. Accepts a non-negative integer value or an enum
   * fl::PaddingMode
   * @param dx dilation of the convolution along the first kernel dimension. A
   *  dilation of 1 is equivalent to a standard convolution along this axis.
   * @param dy dilation of the convolution along the second kernel dimension. A
   *  dilation of 1 is equivalent to a standard convolution along this axis.
   * @param groups the number of groups that the input and output channels
   *  are divided into for restricting the connectivity between input and output
   *  channels. If `groups` > 1, the the output channels in the i-th group will
   *  be only connected to the input channels in the i-th group.
   */
  Conv2D(
      const Variable& w,
      const Variable& b,
      int sx = 1,
      int sy = 1,
      detail::IntOrPadMode px = 0,
      detail::IntOrPadMode py = 0,
      int dx = 1,
      int dy = 1,
      int groups = 1);

  /**
   * Constructs an Conv2D module from another, performing a copy of the
   * parameters.
   *
   * @param other The Conv2D module to copy from.
   */
  Conv2D(const Conv2D& other);

  /**
   * Constructs an Conv2D module from another, performing a copy of the
   * parameters.
   *
   * @param other The Conv2D module to copy from.
   */
  Conv2D& operator=(const Conv2D& other);

  Conv2D(Conv2D&& other) = default;

  Conv2D& operator=(Conv2D&& other) = default;

  Variable forward(const Variable& input) override;

  std::unique_ptr<Module> clone() const override;

  std::string prettyString() const override;

 protected:
  std::shared_ptr<detail::ConvBenchmarks> benchmarks_;
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::Conv2D)
CEREAL_CLASS_VERSION(fl::Conv2D, 1)
