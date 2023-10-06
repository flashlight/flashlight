/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ctc.h> // warpctc

#include "flashlight/pkg/speech/criterion/ConnectionistTemporalClassificationCriterion.h"

#include <flashlight/lib/sequence/criterion/cuda/CriterionUtils.cuh>

#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorBackend.h"

using CriterionUtils = fl::lib::cuda::CriterionUtils<float>;

namespace fl::pkg::speech {

namespace {
inline void throw_on_error(ctcStatus_t status, const char* message) {
  if (status != CTC_STATUS_SUCCESS) {
    throw std::runtime_error(
        message + (", stat = " + std::string(ctcGetStatusString(status))));
  }
}
} // namespace

std::vector<Variable> ConnectionistTemporalClassificationCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& input =
      fl::moddims(inputs[0], {0, 0, 0}); // remove trailing singleton dims
  const auto& target = inputs[1];
  validate(input, target);
  const int N = input.dim(0);
  const int T = input.dim(1);
  const int B = input.dim(2);
  const int batchL = target.dim(0);
  cudaStream_t stream = input.tensor().stream().impl<CUDAStream>().handle();

  ctcOptions options;
  options.loc = CTC_GPU;
  options.stream = stream;
  options.blank_label = N - 1;

  Tensor inputarr({N, B, T}, input.type());
  inputarr(fl::span, fl::span, fl::span) =
      fl::transpose(input.tensor(), {0, 2, 1});

  Tensor grad;
  if (input.isCalcGrad()) {
    grad = fl::full(inputarr.shape(), 0.0, inputarr.type());
  }

  std::vector<int> inputLengths(B, T);
  std::vector<int> labels;
  std::vector<int> labelLengths;
  std::vector<int> batchTargetVec(target.elements());
  target.host(batchTargetVec.data());

  Tensor targetSize({B}, fl::dtype::s32);
  Tensor scale({B}, fl::dtype::f32);

  {
    fl::DevicePtr targetRaw(target.tensor());
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr scaleRaw(scale);

    CriterionUtils::batchTargetSize(
        B,
        batchL,
        batchL,
        static_cast<const int*>(targetRaw.get()),
        static_cast<int*>(targetSizeRaw.get()),
        stream);

    CriterionUtils::computeScale(
        B,
        T,
        N,
        scaleMode_,
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<float*>(scaleRaw.get()),
        stream);
  }

  auto batchTargetSizeVec = targetSize.toHostVector<int>();
  auto batchScaleVec = scale.toHostVector<float>();

  for (int b = 0; b < B; ++b) {
    const int* targetVec = batchTargetVec.data() + b * batchL;
    int L = batchTargetSizeVec[b];

    // A heuristic to modify target length to be able to compute CTC loss
    L = std::min(L, T);
    const int R = fl::pkg::speech::countRepeats(targetVec, L);
    L = std::min(L + R, T) - R;

    labelLengths.push_back(L);
    for (int l = 0; l < L; ++l) {
      labels.push_back(targetVec[l]);
    }
  }
  Tensor batchScales = Tensor::fromVector({B}, batchScaleVec);

  size_t workspace_size;
  throw_on_error(
      get_workspace_size(
          labelLengths.data(),
          inputLengths.data(),
          N,
          B,
          options,
          &workspace_size),
      "Error: get_workspace_size");

  Tensor workspace({static_cast<long long>(workspace_size)}, fl::dtype::b8);

  std::vector<float> costs(B, 0.0);
  {
    DevicePtr inputarrraw(inputarr);
    DevicePtr gradraw(grad);
    DevicePtr workspaceraw(workspace);
    throw_on_error(
        compute_ctc_loss(
            (float*)inputarrraw.get(),
            (float*)gradraw.get(),
            labels.data(),
            labelLengths.data(),
            inputLengths.data(),
            N,
            B,
            costs.data(),
            workspaceraw.get(),
            options),
        "Error: compute_ctc_loss");
  }

  Tensor result = Tensor::fromVector(costs);

  result = result * batchScales;

  auto gradFunc = [grad, batchScales](
                      std::vector<Variable>& moduleInputs,
                      const Variable& grad_output) {
    auto gradScales = grad_output.tensor() * batchScales;
    auto& in = moduleInputs[0];
    gradScales = fl::tile(
        fl::reshape(gradScales, {1, grad_output.dim(0), 1}),
        {in.dim(0), 1, in.dim(1)});
    moduleInputs[0].addGrad(
        Variable(fl::transpose(grad * gradScales, {0, 2, 1}), false));
  };

  return {Variable(result, {input, target}, gradFunc)};
}
} // namespace fl
