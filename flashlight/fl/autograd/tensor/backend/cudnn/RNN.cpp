/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnAutogradExtension.h"

#include <cudnn.h>

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnUtils.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Compute.h"

namespace fl {
namespace {
size_t getWorkspaceSize(
    cudnnHandle_t handle,
    const RNNDescriptor& rnnDesc,
    const int seqLength,
    const TensorDescriptorArray& xDescs) {
  size_t workspaceSize;
  CUDNN_CHECK_ERR(cudnnGetRNNWorkspaceSize(
      handle,
      rnnDesc.descriptor,
      seqLength,
      xDescs.descriptors,
      &workspaceSize));
  return workspaceSize;
}

size_t getReserveSize(
    cudnnHandle_t handle,
    const RNNDescriptor& rnnDesc,
    const int seqLength,
    const TensorDescriptorArray& xDescs) {
  size_t reserveSize;
  CUDNN_CHECK_ERR(cudnnGetRNNTrainingReserveSize(
      handle, rnnDesc.descriptor, seqLength, xDescs.descriptors, &reserveSize));
  return reserveSize;
}

void setCudnnRnnMathType(const Tensor& input, const RNNDescriptor& rnnDesc) {
  if (input.type() == fl::dtype::f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
  }
}

struct CudnnRnnAutogradPayload : public detail::AutogradPayloadData {
  Tensor reserveSpace;
};

} // namespace

std::tuple<Tensor, Tensor, Tensor> CudnnAutogradExtension::rnn(
    const Tensor& input,
    const Tensor& hiddenStateIn,
    const Tensor& cellStateIn,
    const Tensor& weights,
    const int hiddenSize,
    const int numLayers,
    const RnnMode mode,
    const bool bidirectional,
    const float dropProb,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  FL_TENSOR_DTYPES_MATCH_CHECK(input, hiddenStateIn, cellStateIn, weights);

  bool train = (autogradPayload != nullptr);
  auto payload = std::make_shared<CudnnRnnAutogradPayload>();
  if (train) {
    autogradPayload->data = payload;
  }

  Tensor x = input.asContiguousTensor();
  Tensor hiddenState = hiddenStateIn.asContiguousTensor();
  Tensor cellState = cellStateIn.asContiguousTensor();

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);
  setCudnnRnnMathType(input, rnnDesc);

  auto dims = x.shape();
  int inputSize = dims[0];
  int batchSize = dims.ndim() < 2 ? 1 : dims[1];
  int seqLength = dims.ndim() < 3 ? 1 : dims[2];

  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  TensorDescriptorArray xDescs(
      seqLength, x.type(), {1, 1, inputSize, batchSize});

  if (!hiddenState.isEmpty()) {
    auto hxDims = hiddenState.shape();
    int hxHiddenSize = hxDims[0];
    int hxBatchSize = hiddenState.ndim() < 2 ? 1 : hxDims[1];
    int hxTotalLayers = hiddenState.ndim() < 3 ? 1 : hxDims[2];

    if (!(hxHiddenSize == hiddenSize && hxBatchSize == batchSize &&
          hxTotalLayers == totalLayers)) {
      throw std::invalid_argument("invalid hidden state dims for RNN");
    }
  }

  if (!cellState.isEmpty() &&
      !(mode == RnnMode::LSTM && cellState.dim(0) == hiddenSize &&
        cellState.dim(1) == batchSize && cellState.dim(2) == totalLayers)) {
    throw std::invalid_argument("invalid cell state dims for RNN");
  }

  Shape hDims = {1, hiddenSize, batchSize, totalLayers};
  TensorDescriptor hxDesc(x.type(), hDims);
  TensorDescriptor cxDesc(x.type(), hDims);

  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  size_t paramSize;
  CUDNN_CHECK_ERR(cudnnGetRNNParamsSize(
      handle,
      rnnDesc.descriptor,
      xDescs.descriptors[0],
      &paramSize,
      cudnnMapToType(weights.type())));
  if (paramSize != weights.bytes()) {
    throw std::invalid_argument(
        "invalid # of parameters or wrong input shape for RNN");
  }
  FilterDescriptor wDesc(weights);

  Tensor y({outSize, batchSize, seqLength}, input.type());
  TensorDescriptorArray yDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  Tensor hy({hiddenSize, batchSize, totalLayers}, x.type());
  TensorDescriptor hyDesc(x.type(), hDims);

  Tensor cy;
  if (mode == RnnMode::LSTM) {
    cy = Tensor(hy.shape(), x.type());
  }

  TensorDescriptor cyDesc(x.type(), hDims);

  size_t workspaceSize =
    getWorkspaceSize(handle, rnnDesc, seqLength, xDescs);
  size_t reserveSize =
    getReserveSize(handle, rnnDesc, seqLength, xDescs);

  Tensor workspace({static_cast<long long>(workspaceSize)}, fl::dtype::b8);
  // Space must be reused between forward and backward for cuDNN
  payload->reserveSpace =
      Tensor({static_cast<long long>(reserveSize)}, fl::dtype::b8);

  {
    auto contiguousX = x.asContiguousTensor();
    auto contiguousWeights = weights.asContiguousTensor();
    DevicePtr xRaw(contiguousX);
    DevicePtr hxRaw(hiddenState);
    DevicePtr cxRaw(cellState);
    DevicePtr wRaw(contiguousWeights);
    DevicePtr yRaw(y);
    DevicePtr hyRaw(hy);
    DevicePtr cyRaw(cy);
    DevicePtr workspaceRaw(workspace);
    DevicePtr reserveSpaceRaw(payload->reserveSpace);
    // ensure cudnn compute stream waits on input/output tensor streams
    relativeSync(cudnnStream, {
      contiguousX, hiddenState, cellState, contiguousWeights, y, hy, cy,
      workspace, payload->reserveSpace,
    });

    CUDNN_CHECK_ERR(cudnnRNNForwardTraining(
        handle,
        rnnDesc.descriptor,
        seqLength,
        xDescs.descriptors,
        xRaw.get(),
        hxDesc.descriptor,
        hxRaw.get(),
        cxDesc.descriptor,
        cxRaw.get(),
        wDesc.descriptor,
        wRaw.get(),
        yDesc.descriptors,
        yRaw.get(),
        hyDesc.descriptor,
        hyRaw.get(),
        cyDesc.descriptor,
        cyRaw.get(),
        workspaceRaw.get(),
        workspaceSize,
        reserveSpaceRaw.get(),
        reserveSize));
  }

  // ensure output tensor streams wait on cudnn compute stream
  relativeSync({y, hy, cy}, cudnnStream);
  return std::make_tuple(y, hy, cy);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> CudnnAutogradExtension::rnnBackward(
    const Tensor& input,
    const Tensor& hiddenState,
    const Tensor& cellState,
    const Tensor& weights,
    const std::shared_ptr<detail::RNNGradData> gradData,
    const Tensor& output,
    const int numLayers,
    const int hiddenSize,
    const RnnMode mode,
    const bool bidirectional,
    const float dropProb,
    std::shared_ptr<detail::AutogradPayload> autogradPayload) {
  if (!autogradPayload) {
    throw std::invalid_argument(
        "CudnnAutogradExtension::rnnBackward given null detail::AutogradPayload");
  }
  auto payload =
      std::static_pointer_cast<CudnnRnnAutogradPayload>(autogradPayload->data);

  auto handle = getCudnnHandle();
  const auto& cudnnStream = getCudnnStream();

  auto x = input.asContiguousTensor();
  auto& y = output;

  auto dims = x.shape();
  int inputSize = dims[0];
  int batchSize = dims.ndim() < 2 ? 1 : dims[1];
  int seqLength = dims.ndim() < 3 ? 1 : dims[2];
  int totalLayers = numLayers * (bidirectional ? 2 : 1);
  int outSize = hiddenSize * (bidirectional ? 2 : 1);

  DropoutDescriptor dropout(dropProb);
  RNNDescriptor rnnDesc(
      input.type(), hiddenSize, numLayers, mode, bidirectional, dropout);
  setCudnnRnnMathType(input, rnnDesc);

  TensorDescriptorArray yDesc(seqLength, y.type(), {1, 1, outSize, batchSize});
  TensorDescriptorArray dyDesc(seqLength, y.type(), {1, 1, outSize, batchSize});

  Shape hDims = {1, hiddenSize, batchSize, totalLayers};
  TensorDescriptor dhyDesc(x.type(), hDims);
  TensorDescriptor dcyDesc(x.type(), hDims);
  TensorDescriptor hxDesc(x.type(), hDims);
  TensorDescriptor cxDesc(x.type(), hDims);

  Tensor dhx(hiddenState.shape(), hiddenState.type());
  Tensor dcx(cellState.shape(), cellState.type());
  TensorDescriptor dhxDesc(x.type(), hDims);
  TensorDescriptor dcxDesc(x.type(), hDims);

  FilterDescriptor wDesc(weights);

  Tensor dx(input.shape(), input.type());
  TensorDescriptorArray dxDescs(
      seqLength, dx.type(), {1, 1, inputSize, batchSize});

  size_t workspaceSize =
    getWorkspaceSize(handle, rnnDesc, seqLength, dxDescs);
  Tensor workspace({static_cast<long long>(workspaceSize)}, fl::dtype::b8);

  auto& dy = gradData->dy;
  if (dy.isEmpty()) {
    dy = fl::full(y.shape(), 0.0, y.type());
  }
  auto& dhy = gradData->dhy;
  auto& dcy = gradData->dcy;

  DevicePtr yRaw(output);
  DevicePtr workspaceRaw(workspace);
  DevicePtr reserveSpaceRaw(payload->reserveSpace);
  // ensure cudnn compute stream waits on input/output tensor streams
  relativeSync(cudnnStream, {output, workspace, payload->reserveSpace});

  {
    DevicePtr dyRaw(dy); // Has to be set to 0 if empty
    DevicePtr dhyRaw(dhy);
    DevicePtr dcyRaw(dcy);

    DevicePtr wRaw(weights);

    DevicePtr hxRaw(hiddenState);
    DevicePtr cxRaw(cellState);

    DevicePtr dxRaw(dx);
    DevicePtr dhxRaw(dhx);
    DevicePtr dcxRaw(dcx);
    // ensure cudnn compute stream waits on input/output tensor streams
    relativeSync(
        cudnnStream,
        {dy, dhy, dcy, weights, hiddenState, cellState, dx, dhx, dcx});

    /* We need to update reserveSpace even if we just want the
     * weight gradients. */
    CUDNN_CHECK_ERR(cudnnRNNBackwardData(
        handle,
        rnnDesc.descriptor,
        seqLength,
        yDesc.descriptors,
        yRaw.get(),
        dyDesc.descriptors,
        dyRaw.get(),
        dhyDesc.descriptor,
        dhyRaw.get(),
        dcyDesc.descriptor,
        dcyRaw.get(),
        wDesc.descriptor,
        wRaw.get(),
        hxDesc.descriptor,
        hxRaw.get(),
        cxDesc.descriptor,
        cxRaw.get(),
        dxDescs.descriptors,
        dxRaw.get(),
        dhxDesc.descriptor,
        dhxRaw.get(),
        dcxDesc.descriptor,
        dcxRaw.get(),
        workspaceRaw.get(),
        workspaceSize,
        reserveSpaceRaw.get(),
        payload->reserveSpace.bytes()));
  }

  if (input.type() == fl::dtype::f16) {
    CUDNN_CHECK_ERR(cudnnSetRNNMatrixMathType(
        rnnDesc.descriptor, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));
  } else {
    CUDNN_CHECK_ERR(
        cudnnSetRNNMatrixMathType(rnnDesc.descriptor, CUDNN_DEFAULT_MATH));
  }
  TensorDescriptorArray xDescs(
      seqLength, x.type(), {1, 1, inputSize, batchSize});
  Tensor dw = fl::full(weights.shape(), 0, weights.type());

  FilterDescriptor dwDesc(dw);

  {
    DevicePtr xRaw(x);
    DevicePtr dwRaw(dw);
    DevicePtr hxRaw(hiddenState);
    // ensure cudnn compute stream waits on input/output tensor streams
    relativeSync(cudnnStream, {x, dw, hiddenState});

    CUDNN_CHECK_ERR(cudnnRNNBackwardWeights(
        handle,
        rnnDesc.descriptor,
        seqLength,
        xDescs.descriptors,
        xRaw.get(),
        hxDesc.descriptor,
        hxRaw.get(),
        yDesc.descriptors,
        yRaw.get(),
        workspaceRaw.get(),
        workspaceSize,
        dwDesc.descriptor,
        dwRaw.get(),
        reserveSpaceRaw.get(),
        payload->reserveSpace.bytes()));
  }

  // ensure output tensor streams wait on cudnn compute stream
  relativeSync({dx, dhx, dcx, dw}, cudnnStream);
  return std::make_tuple(dx, dhx, dcx, dw);
}

} // namespace fl
