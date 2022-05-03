/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/attention/Utils.h"
#include "flashlight/pkg/speech/criterion/attention/Defines.h"

#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/tensor/Index.h"

namespace fl {
namespace pkg {
namespace speech {

Variable maskAttention(const Variable& input, const Variable& sizes) {
  int B = input.dims(2);
  int T = input.dims(1);
  // xEncodedSizes is (1, B) size
  Tensor inputNotPaddedSize =
      fl::ceil(sizes.tensor() / fl::amax(sizes.tensor()).asScalar<float>() * T);
  Tensor padMask =
      fl::iota({T, 1}, {1, B}) >= fl::tile(inputNotPaddedSize, {T, 1});
  padMask = fl::tile(fl::reshape(padMask, {1, T, B}), {input.dims(0), 1, 1});

  Tensor output = input.tensor();
  output(padMask) = kAttentionMaskValue;

  auto gradFunc =
      [padMask](std::vector<Variable>& inputs, const Variable& gradOutput) {
        Tensor gradArray = gradOutput.tensor();
        gradArray(padMask) = 0.;
        inputs[0].addGrad(Variable(gradArray, false));
      };
  return Variable(output, {input.withoutData()}, gradFunc);
}

} // namespace speech
} // namespace pkg
} // namespace fl
