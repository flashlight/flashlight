/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/cudnn/CudnnUtils.h"

#include <array>
#include <stdexcept>
#include <unordered_map>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace {

struct DeviceHandle {
  cudnnHandle_t cudnnHandle;
  std::shared_ptr<fl::CUDAStream> stream;

  explicit DeviceHandle(std::shared_ptr<fl::CUDAStream> _stream)
    : cudnnHandle(nullptr), stream(_stream) {
    CUDNN_CHECK_ERR(cudnnCreate(&cudnnHandle));
    CUDNN_CHECK_ERR(cudnnSetStream(cudnnHandle, stream->handle()));
  }

  ~DeviceHandle() {
    if (cudnnHandle) {
// See https://git.io/fNQnM - sometimes, at exit, the CUDA context
// (or something) is already destroyed by the time a handle gets destroyed
// because of an issue with the destruction order.
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
      CUDNN_CHECK_ERR(cudnnDestroy(cudnnHandle));
#endif
    }
  }
};

const float kFloatZero = 0.0;
const float kFloatOne = 1.0;

const double kDoubleZero = 0.0;
const double kDoubleOne = 1.0;

// TODO: move this to CudnnAutogradExtension if we make it a singleton
std::unordered_map<int, DeviceHandle> handles;

const DeviceHandle& getActiveDeviceHandle() {
  auto& manager = fl::DeviceManager::getInstance();
  auto& cudaDevice =
    manager.getActiveDevice(fl::DeviceType::CUDA).impl<fl::CUDADevice>();
  int id = cudaDevice.nativeId();
  // lazily initialize cuda stream for cudnn
  if (handles.count(id) == 0) {
#ifdef NO_CUDNN_DESTROY_HANDLE
    // NOTE unmanaged so to avoid CUDA driver shut down prior to stream
    // destruction. This is safe because this object is always part of a global
    // map -- the resource won't be relased until program shutdown anyway.
    auto stream = fl::CUDAStream::createUnmanaged();
#else
    auto stream = fl::CUDAStream::createManaged();
#endif
    handles.emplace(id, DeviceHandle(stream));
  }
  return handles.at(id);
}

// See https://git.io/fp9oo for an explanation.
#if CUDNN_VERSION < 7000
struct CudnnDropoutStruct {
  float dropout;
  int nstates;
  void* states;
};
#endif

} // namespace

namespace fl {

void cudnnCheckErr(cudnnStatus_t status) {
  if (status == CUDNN_STATUS_SUCCESS) {
    return;
  }
  const char* err = cudnnGetErrorString(status);
  switch (status) {
    case CUDNN_STATUS_BAD_PARAM:
      throw std::invalid_argument(err);
    default:
      throw std::runtime_error(err);
  }
}

cudnnDataType_t cudnnMapToType(const fl::dtype& t) {
  switch (t) {
    case fl::dtype::f16:
      return CUDNN_DATA_HALF;
    case fl::dtype::f32:
      return CUDNN_DATA_FLOAT;
    case fl::dtype::f64:
      return CUDNN_DATA_DOUBLE;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

cudnnPoolingMode_t cudnnMapToPoolingMode(const PoolingMode mode) {
  switch (mode) {
    case PoolingMode::MAX:
      return CUDNN_POOLING_MAX;
    case PoolingMode::AVG_INCLUDE_PADDING:
      return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    case PoolingMode::AVG_EXCLUDE_PADDING:
      return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    default:
      throw std::invalid_argument("unsupported pooling mode for cuDNN");
  }
}

cudnnRNNMode_t cudnnMapToRNNMode(const RnnMode mode) {
  switch (mode) {
    case RnnMode::RELU:
      return CUDNN_RNN_RELU;
    case RnnMode::TANH:
      return CUDNN_RNN_TANH;
    case RnnMode::LSTM:
      return CUDNN_LSTM;
    case RnnMode::GRU:
      return CUDNN_GRU;
    default:
      throw std::invalid_argument("unsupported RNN mode for cuDNN");
  }
}

TensorDescriptor::TensorDescriptor(const fl::dtype type, const Shape& flDims) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(type);

  std::array<int, 4> dims = {1, 1, 1, 1};
  // We want, if dims exist:
  // {flDims[3], flDims[2], flDims[1], flDims[0]};
  for (unsigned i = 0; i < flDims.ndim(); ++i) {
    dims[3 - i] = flDims[i];
  }

  // Sets strides so array is contiguous row-major for cudnn
  std::vector<int> r_strides = {1};
  for (auto it = dims.rbegin(); it != dims.rend() - 1; ++it) {
    r_strides.push_back(r_strides.back() * (*it));
  }
  std::vector<int> strides(r_strides.rbegin(), r_strides.rend());

  CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
      descriptor, cudnntype, dims.size(), dims.data(), strides.data()));
}

TensorDescriptor::TensorDescriptor(const Tensor& input) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(input.type());

  auto flStrides = input.strides();
  auto flDims = input.shape();

  // reverse the dims (column -> row major) and cast to int type
  std::array<int, 4> strides = {1, 1, 1, 1};
  //  {flStrides[3], flStrides[2], flStrides[1], flStrides[0]};
  for (unsigned i = 0; i < flStrides.ndim(); ++i) {
    strides[3 - i] = flStrides[i];
  }

  std::array<int, 4> dims = {1, 1, 1, 1};
  // {flDims[3], flDims[2], flDims[1], flDims[0]};
  for (unsigned i = 0; i < flDims.ndim(); ++i) {
    dims[3 - i] = flDims[i];
  }

  CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
      descriptor /* descriptor handle */,
      cudnntype /* = dataType */,
      4,
      dims.data(),
      strides.data()));
}

TensorDescriptor::~TensorDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyTensorDescriptor(descriptor));
}

TensorDescriptorArray::TensorDescriptorArray(
    int size,
    const fl::dtype type,
    const Shape& dims) {
  desc_vec.reserve(size);
  for (int i = 0; i < size; i++) {
    desc_vec.emplace_back(type, dims);
    desc_raw_vec.push_back(desc_vec.back().descriptor);
  }
  descriptors = desc_raw_vec.data();
}

TensorDescriptorArray::~TensorDescriptorArray() = default;

PoolingDescriptor::PoolingDescriptor(
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py,
    PoolingMode mode) {
  CUDNN_CHECK_ERR(cudnnCreatePoolingDescriptor(&descriptor));
  std::array<int, 2> window = {(int)wy, (int)wx};
  std::array<int, 2> padding = {(int)py, (int)px};
  std::array<int, 2> stride = {(int)sy, (int)sx};

  auto cudnnpoolingmode = cudnnMapToPoolingMode(mode);
  CUDNN_CHECK_ERR(cudnnSetPoolingNdDescriptor(
      descriptor,
      cudnnpoolingmode,
      CUDNN_PROPAGATE_NAN,
      2,
      window.data(),
      padding.data(),
      stride.data()));
}

PoolingDescriptor::~PoolingDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyPoolingDescriptor(descriptor));
}

FilterDescriptor::FilterDescriptor(const Tensor& input) {
  CUDNN_CHECK_ERR(cudnnCreateFilterDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(input.type());

  auto flDims = input.shape();
  std::array<int, 4> dims = {1, 1, 1, 1};
  // We want, if dims exist:
  // {flDims[3], flDims[2], flDims[1], flDims[0]};
  for (unsigned i = 0; i < flDims.ndim(); ++i) {
    dims[3 - i] = flDims[i];
  }

  CUDNN_CHECK_ERR(cudnnSetFilterNdDescriptor(
      descriptor, cudnntype, CUDNN_TENSOR_NCHW, 4, dims.data()));
}

FilterDescriptor::~FilterDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyFilterDescriptor(descriptor));
}

DropoutDescriptor::DropoutDescriptor(float drop_prob) {
  CUDNN_CHECK_ERR(cudnnCreateDropoutDescriptor(&descriptor));
  auto cudnnHandle = getCudnnHandle();
  unsigned long long seed = 0;
  size_t state_size;
  CUDNN_CHECK_ERR(cudnnDropoutGetStatesSize(cudnnHandle, &state_size));
  auto& dropout_states = getDropoutStates();
  if (dropout_states.isEmpty()) {
    dropout_states =
        Tensor({static_cast<long long>(state_size)}, fl::dtype::b8);
    DevicePtr statesraw(dropout_states);
    CUDNN_CHECK_ERR(cudnnSetDropoutDescriptor(
        descriptor, cudnnHandle, drop_prob, statesraw.get(), state_size, seed));
  } else {
    DevicePtr statesraw(dropout_states);
// See https://git.io/fp9oo for an explanation.
#if CUDNN_VERSION >= 7000
    CUDNN_CHECK_ERR(cudnnRestoreDropoutDescriptor(
        descriptor, cudnnHandle, drop_prob, statesraw.get(), state_size, seed));
#else
    auto dropout_struct = reinterpret_cast<CudnnDropoutStruct*>(descriptor);
    dropout_struct->dropout = drop_prob;
    dropout_struct->nstates = state_size;
    dropout_struct->states = statesraw.get();
#endif
  }
}

DropoutDescriptor::~DropoutDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyDropoutDescriptor(descriptor));
}

Tensor& DropoutDescriptor::getDropoutStates() {
  thread_local Tensor dropout_states;
  return dropout_states;
}

RNNDescriptor::RNNDescriptor(
    fl::dtype type,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional,
    DropoutDescriptor& dropout) {
  CUDNN_CHECK_ERR(cudnnCreateRNNDescriptor(&descriptor));

  auto cudnnHandle = getCudnnHandle();

  cudnnRNNInputMode_t in_mode = CUDNN_LINEAR_INPUT;

  cudnnDirectionMode_t dir =
      bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;

  cudnnRNNMode_t cell = cudnnMapToRNNMode(mode);
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  cudnnDataType_t cudnntype = cudnnMapToType(type);

#if CUDNN_VERSION >= 7000 && CUDNN_VERSION < 8000
  CUDNN_CHECK_ERR(cudnnSetRNNDescriptor(
      cudnnHandle,
      descriptor,
      hidden_size,
      num_layers,
      dropout.descriptor,
      in_mode,
      dir,
      cell,
      algo,
      cudnntype));
#else
  CUDNN_CHECK_ERR(cudnnSetRNNDescriptor_v6(
      cudnnHandle,
      descriptor,
      hidden_size,
      num_layers,
      dropout.descriptor,
      in_mode,
      dir,
      cell,
      algo,
      cudnntype));
#endif
}

RNNDescriptor::~RNNDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyRNNDescriptor(descriptor));
}

ConvDescriptor::ConvDescriptor(
    fl::dtype type,
    int px,
    int py,
    int sx,
    int sy,
    int dx,
    int dy,
    int groups) {
  CUDNN_CHECK_ERR(cudnnCreateConvolutionDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(type);
  std::array<int, 2> padding = {(int)py, (int)px};
  std::array<int, 2> stride = {(int)sy, (int)sx};
  std::array<int, 2> dilation = {(int)dy, (int)dx};

  CUDNN_CHECK_ERR(cudnnSetConvolutionNdDescriptor(
      descriptor,
      2,
      padding.data(),
      stride.data(),
      dilation.data(),
      CUDNN_CROSS_CORRELATION,
      cudnntype));

  CUDNN_CHECK_ERR(cudnnSetConvolutionGroupCount(descriptor, groups));
}

ConvDescriptor::~ConvDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyConvolutionDescriptor(descriptor));
}

cudnnHandle_t getCudnnHandle() {
  return getActiveDeviceHandle().cudnnHandle;
}

const CUDAStream& getCudnnStream() {
  return *getActiveDeviceHandle().stream;
}

const void* kOne(const fl::dtype t) {
  switch (t) {
    case fl::dtype::f16:
    case fl::dtype::f32:
      return &kFloatOne;
    case fl::dtype::f64:
      return &kDoubleOne;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

const void* kZero(const fl::dtype t) {
  switch (t) {
    case fl::dtype::f16:
    case fl::dtype::f32:
      return &kFloatZero;
    case fl::dtype::f64:
      return &kDoubleZero;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

} // namespace fl
