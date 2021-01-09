/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/app/asr/criterion/attention/Utils.h"

#include "flashlight/fl/flashlight.h"

namespace fl {
namespace app {
namespace asr {

Variable maskAttention(const Variable& input, const Variable& sizes) {
  int B = input.dims(2);
  int T = input.dims(1);
  // xEncodedSizes is (1, B) size
  af::array inputNotPaddedSize =
      af::ceil(sizes.array() / af::max<float>(sizes.array()) * T);
  auto padMask = af::iota(af::dim4(T, 1), af::dim4(1, B)) >=
      af::tile(inputNotPaddedSize, T, 1);
  padMask =
      af::tile(af::moddims(padMask, af::dim4(1, T, B)), input.dims(0), 1, 1);

  af::array output = af::flat(input.array());
  af::array flatPadMask = af::flat(padMask);
  auto inputDims = input.dims();

  output(flatPadMask) = kAttentionMaskValue;
  output = af::moddims(output, inputDims);

  auto gradFunc = [flatPadMask, inputDims](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    af::array gradArray = af::flat(gradOutput.array());
    gradArray(flatPadMask) = 0.;
    auto grad = Variable(af::moddims(gradArray, inputDims), false);
    inputs[0].addGrad(grad);
  };
  return Variable(output, {input.withoutData()}, gradFunc);
}
} // namespace asr
} // namespace app
} // namespace fl