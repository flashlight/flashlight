/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/backend/cuda/CudnnUtils.h"

#include <array>
#include <stdexcept>
#include <unordered_map>

#include <af/internal.h>

#include "flashlight/fl/common/backend/cuda/CudaUtils.h"
#include "flashlight/fl/common/DevicePtr.h"

namespace {

struct Handle {
  cudnnHandle_t handle;
  Handle() : handle(nullptr) {
    CUDNN_CHECK_ERR(cudnnCreate(&handle));
    CUDNN_CHECK_ERR(cudnnSetStream(handle, fl::cuda::getActiveStream()));
  }
  ~Handle() {
    if (handle) {
// See https://git.io/fNQnM - sometimes, at exit, the CUDA context
// (or something) is already destroyed by the time a handle gets destroyed
// because of an issue with the destruction order.
#ifdef NO_CUDNN_DESTROY_HANDLE
#else
      CUDNN_CHECK_ERR(cudnnDestroy(handle));
#endif
    }
  }
};

const float kFloatZero = 0.0;
const float kFloatOne = 1.0;

const double kDoubleZero = 0.0;
const double kDoubleOne = 1.0;

std::unordered_map<int, Handle> handles;

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

cudnnDataType_t cudnnMapToType(const af::dtype& t) {
  switch (t) {
    case af::dtype::f16:
      return CUDNN_DATA_HALF;
    case af::dtype::f32:
      return CUDNN_DATA_FLOAT;
    case af::dtype::f64:
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

TensorDescriptor::TensorDescriptor(const Variable& input)
    : TensorDescriptor(input.array()) {}

TensorDescriptor::TensorDescriptor(
    const af::dtype type,
    const af::dim4& af_dims) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(type);

  std::array<int, 4> dims = {
      (int)af_dims[3], (int)af_dims[2], (int)af_dims[1], (int)af_dims[0]};

  // Sets strides so array is contiguous row-major for cudnn
  std::vector<int> r_strides = {1};
  for (auto it = dims.rbegin(); it != dims.rend() - 1; ++it) {
    r_strides.push_back(r_strides.back() * (*it));
  }
  std::vector<int> strides(r_strides.rbegin(), r_strides.rend());

  CUDNN_CHECK_ERR(cudnnSetTensorNdDescriptor(
      descriptor, cudnntype, dims.size(), dims.data(), strides.data()));
}

TensorDescriptor::TensorDescriptor(const af::array& input) {
  CUDNN_CHECK_ERR(cudnnCreateTensorDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(input.type());

  auto afstrides = af::getStrides(input);
  auto afdims = input.dims();

  // reverse the arrays and cast to int type
  std::array<int, 4> strides = {(int)afstrides[3],
                                (int)afstrides[2],
                                (int)afstrides[1],
                                (int)afstrides[0]};
  std::array<int, 4> dims = {
      (int)afdims[3], (int)afdims[2], (int)afdims[1], (int)afdims[0]};

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
    const af::dtype type,
    const af::dim4& dims) {
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

FilterDescriptor::FilterDescriptor(const Variable& input)
    : FilterDescriptor(input.array()) {}

FilterDescriptor::FilterDescriptor(const af::array& input) {
  CUDNN_CHECK_ERR(cudnnCreateFilterDescriptor(&descriptor));
  cudnnDataType_t cudnntype = cudnnMapToType(input.type());
  auto afdims = input.dims();
  std::array<int, 4> dims = {
      (int)afdims[3], (int)afdims[2], (int)afdims[1], (int)afdims[0]};

  CUDNN_CHECK_ERR(cudnnSetFilterNdDescriptor(
      descriptor, cudnntype, CUDNN_TENSOR_NCHW, 4, dims.data()));
}

FilterDescriptor::~FilterDescriptor() {
  CUDNN_CHECK_ERR(cudnnDestroyFilterDescriptor(descriptor));
}

DropoutDescriptor::DropoutDescriptor(float drop_prob) {
  CUDNN_CHECK_ERR(cudnnCreateDropoutDescriptor(&descriptor));
  auto handle = getCudnnHandle();
  unsigned long long seed = 0;
  size_t state_size;
  CUDNN_CHECK_ERR(cudnnDropoutGetStatesSize(handle, &state_size));
  auto& dropout_states = getDropoutStates();
  if (dropout_states.isempty()) {
    dropout_states = af::array(state_size, af::dtype::b8);
    DevicePtr statesraw(dropout_states);
    CUDNN_CHECK_ERR(cudnnSetDropoutDescriptor(
        descriptor, handle, drop_prob, statesraw.get(), state_size, seed));
  } else {
    DevicePtr statesraw(dropout_states);
// See https://git.io/fp9oo for an explanation.
#if CUDNN_VERSION >= 7000
    CUDNN_CHECK_ERR(cudnnRestoreDropoutDescriptor(
        descriptor, handle, drop_prob, statesraw.get(), state_size, seed));
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

af::array& DropoutDescriptor::getDropoutStates() {
  thread_local af::array dropout_states;
  return dropout_states;
}

RNNDescriptor::RNNDescriptor(
    af::dtype type,
    int hidden_size,
    int num_layers,
    RnnMode mode,
    bool bidirectional,
    DropoutDescriptor& dropout) {
  CUDNN_CHECK_ERR(cudnnCreateRNNDescriptor(&descriptor));

  auto handle = getCudnnHandle();

  cudnnRNNInputMode_t in_mode = CUDNN_LINEAR_INPUT;

  cudnnDirectionMode_t dir =
      bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;

  cudnnRNNMode_t cell = cudnnMapToRNNMode(mode);
  cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
  cudnnDataType_t cudnntype = cudnnMapToType(type);

#if CUDNN_VERSION >= 7000 && CUDNN_VERSION < 8000 
  CUDNN_CHECK_ERR(cudnnSetRNNDescriptor(
      handle,
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
      handle,
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
    af::dtype type,
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
  int af_id = af::getDevice();
  return handles[af_id].handle;
}

const void* kOne(const af::dtype t) {
  switch (t) {
    case af::dtype::f16:
    case af::dtype::f32:
      return &kFloatOne;
    case af::dtype::f64:
      return &kDoubleOne;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

const void* kZero(const af::dtype t) {
  switch (t) {
    case af::dtype::f16:
    case af::dtype::f32:
      return &kFloatZero;
    case af::dtype::f64:
      return &kDoubleZero;
    default:
      throw std::invalid_argument("unsupported data type for cuDNN");
  }
}

} // namespace fl
