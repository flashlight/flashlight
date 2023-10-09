/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/speech/criterion/ForceAlignmentCriterion.h"

#include <stdexcept>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include <flashlight/lib/sequence/criterion/cuda/ForceAlignmentCriterion.cuh>

using FAC = fl::lib::cuda::ForceAlignmentCriterion<float>;

namespace fl::pkg::speech {

static void backward(
    std::vector<Variable>& inputs,
    const Variable& gradVar,
    int B,
    int T,
    int N,
    int L,
    const Tensor& target,
    const Tensor& targetSize,
    Tensor& workspace) {
  if (gradVar.type() != fl::dtype::f32) {
    throw std::invalid_argument("FAC: grad must be float32");
  }
  if (inputs.size() != 2) {
    throw std::invalid_argument(
        "ForceAlignmentCriterion backward expects two input args");
  }

  const auto& grad = gradVar.tensor();
  Tensor inputGrad({N, T, B}, fl::dtype::f32);
  Tensor transGrad({N, N}, fl::dtype::f32);

  {
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr gradRaw(grad);
    fl::DevicePtr inputGradRaw(inputGrad);
    fl::DevicePtr transGradRaw(transGrad);
    fl::DevicePtr workspaceRaw(workspace);
    FAC::backward(
        B,
        T,
        N,
        L,
        static_cast<const int*>(targetRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(gradRaw.get()),
        static_cast<float*>(inputGradRaw.get()),
        static_cast<float*>(transGradRaw.get()),
        workspaceRaw.get(),
        inputs[0].tensor().stream().impl<CUDAStream>().handle());
  }

  inputs[0].addGrad(Variable(inputGrad, false));
  inputs[1].addGrad(Variable(transGrad, false));
}

Variable ForceAlignmentCriterion::forward(
    const Variable& inputVar,
    const Variable& targetVar) {
  const auto& transVar = param(0);
  int B = inputVar.dim(2);
  int T = inputVar.dim(1);
  int N = inputVar.dim(0);
  int L = targetVar.dim(0);

  if (N != transVar.dim(0)) {
    throw std::invalid_argument("FAC: input dim doesn't match N");
  } else if (inputVar.type() != fl::dtype::f32) {
    throw std::invalid_argument("FAC: input must be float32");
  } else if (targetVar.type() != fl::dtype::s32) {
    throw std::invalid_argument("FAC: target must be int32");
  }

  const auto& input = inputVar.tensor();
  const auto& target = targetVar.tensor();
  const auto& targetSize = getTargetSizeArray(target, T);
  const auto& trans = transVar.tensor();
  Tensor loss({B}, fl::dtype::f32);
  Tensor workspace(
      {static_cast<long long>(FAC::getWorkspaceSize(B, T, N, L))},
      fl::dtype::u8);

  {
    fl::DevicePtr inputRaw(input);
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr lossRaw(loss);
    fl::DevicePtr workspaceRaw(workspace);

    FAC::forward(
        B,
        T,
        N,
        L,
        scaleMode_,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const int*>(targetRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<float*>(lossRaw.get()),
        workspaceRaw.get(),
        input.stream().impl<CUDAStream>().handle());
  }

  return Variable(
      loss,
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) mutable {
        backward(inputs, gradVar, B, T, N, L, target, targetSize, workspace);
      });
}

Tensor ForceAlignmentCriterion::viterbiPath(
    const Tensor& input,
    const Tensor& target) {
  if (input.ndim() != 3) {
    throw std::invalid_argument(
        "ForceAlignmentCriterion::viterbiPath: "
        "expects input with dimensions {N, T, B}");
  }
  int N = input.dim(0);
  int T = input.dim(1);
  int B = input.dim(2);
  int L = target.dim(0);

  std::vector<std::vector<int>> bestPaths;
  const auto& transVar = param(0);

  if (N != transVar.dim(0)) {
    throw std::invalid_argument("FAC: input dim doesn't match N:");
  } else if (input.type() != fl::dtype::f32) {
    throw std::invalid_argument("FAC: input must be float32");
  } else if (target.type() != fl::dtype::s32) {
    throw std::invalid_argument("FAC: target must be int32");
  }

  const auto& targetSize = getTargetSizeArray(target, T);
  const auto& trans = transVar.tensor();
  Tensor bestPathsVar({T, B}, fl::dtype::s32);
  Tensor workspace(
      {static_cast<long long>(FAC::getWorkspaceSize(B, T, N, L))},
      fl::dtype::u8);

  {
    fl::DevicePtr inputRaw(input);
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr bestPathsRaw(bestPathsVar);
    ;
    fl::DevicePtr workspaceRaw(workspace);

    FAC::viterbiPath(
        B,
        T,
        N,
        L,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const int*>(targetRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<int*>(bestPathsRaw.get()),
        workspaceRaw.get(),
        input.stream().impl<CUDAStream>().handle());
  }
  return bestPathsVar;
}
} // namespace fl
