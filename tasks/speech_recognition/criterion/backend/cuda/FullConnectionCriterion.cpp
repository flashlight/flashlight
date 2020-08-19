/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/FullConnectionCriterion.h"

#include <flashlight/common/cuda.h>

#include "criterion/CriterionUtils.h"
#include "libraries/criterion/cuda/FullConnectionCriterion.cuh"

using fl::Variable;
using FCC = w2l::cuda::FullConnectionCriterion<float>;

namespace w2l {

static void backward(
    std::vector<Variable>& inputs,
    const Variable& gradVar,
    int B,
    int T,
    int N,
    const af::array& trans,
    af::array& workspace) {
  if (gradVar.type() != f32) {
    throw std::invalid_argument("FCC: grad must be float32");
  }

  const auto& grad = gradVar.array();
  af::array inputGrad(N, T, B, f32);
  af::array transGrad(N, N, f32);

  {
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr gradRaw(grad);
    fl::DevicePtr inputGradRaw(inputGrad);
    fl::DevicePtr transGradRaw(transGrad);
    fl::DevicePtr workspaceRaw(workspace);
    FCC::backward(
        B,
        T,
        N,
        static_cast<const float*>(transRaw.get()),
        static_cast<const float*>(gradRaw.get()),
        static_cast<float*>(inputGradRaw.get()),
        static_cast<float*>(transGradRaw.get()),
        workspaceRaw.get(),
        fl::cuda::getActiveStream());
  }

  inputs[0].addGrad(Variable(inputGrad, false));
  inputs[1].addGrad(Variable(transGrad, false));
}

Variable FullConnectionCriterion::forward(
    const Variable& inputVar,
    const Variable& targetVar) {
  const auto& transVar = param(0);
  int B = inputVar.dims(2);
  int T = inputVar.dims(1);
  int N = inputVar.dims(0);

  if (N != transVar.dims(0)) {
    throw std::invalid_argument("FCC: input dim doesn't match N");
  } else if (inputVar.type() != f32) {
    throw std::invalid_argument("FCC: input must be float32");
  } else if (targetVar.type() != s32) {
    throw std::invalid_argument("FCC: target must be int32");
  }

  const auto& input = inputVar.array();
  const auto& target = targetVar.array();
  const auto& targetSize = getTargetSizeArray(target, T);
  const auto& trans = transVar.array();
  af::array loss(B, f32);
  af::array workspace(FCC::getWorkspaceSize(B, T, N), u8);

  {
    fl::DevicePtr inputRaw(input);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr lossRaw(loss);
    fl::DevicePtr workspaceRaw(workspace);

    FCC::forward(
        B,
        T,
        N,
        scaleMode_,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<float*>(lossRaw.get()),
        workspaceRaw.get(),
        fl::cuda::getActiveStream());
  }

  return Variable(
      loss,
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) mutable {
        backward(inputs, gradVar, B, T, N, trans, workspace);
      });
}

} // namespace w2l
