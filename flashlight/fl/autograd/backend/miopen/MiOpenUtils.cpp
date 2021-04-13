/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/backend/miopen/MiOpenUtils.h"

#include <array>
#include <stdexcept>
#include <unordered_map>

#include <af/internal.h>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/common/OpenClUtils.h"
#include "flashlight/fl/common/backend/miopen/MiOpenUtils.h"

namespace {

struct Handle {
  miopenHandle_t handle;
  Handle() : handle(nullptr) {
    MIOPEN_CHECK_ERR(miopenCreateWithStream(
        &handle, static_cast<miopenAcceleratorQueue_t>(fl::ocl::getQueue())));
  }
  ~Handle() {
    MIOPEN_CHECK_ERR(miopenDestroy(handle));
  }
};

const float kFloatZero = 0.0;
const float kFloatOne = 1.0;

std::unordered_map<int, Handle> handles;

miopenTensorDescriptor_t createAndSetTensorDescriptor(const af::array& input) {
  miopenTensorDescriptor_t descriptor;
  MIOPEN_CHECK_ERR(miopenCreateTensorDescriptor(&descriptor));
  miopenDataType_t miopentype = fl::miopenMapToType(input.type());

  const auto afstrides = af::getStrides(input);
  const auto afdims = input.dims();

  // reverse the arrays and cast to int type
  std::vector<int> strides(afstrides.elements());
  std::vector<int> dims(afdims.elements());

  for (int i = 0; i < afstrides.elements(); ++i) {
    strides[i] = static_cast<int>(afstrides[afstrides.elements() - i - 1]);
  }

  for (int j = 0; j < afstrides.elements(); ++j) {
    dims[j] = static_cast<int>(afdims[afdims.elements() - j - 1]);
  }

  MIOPEN_CHECK_ERR(miopenSetTensorDescriptor(
      descriptor /* descriptor handle */,
      miopentype /* = dataType */,
      /* nbDims= */ 4,
      dims.data(),
      strides.data()));

  return descriptor;
}

} // namespace

namespace fl {

miopenDataType_t miopenMapToType(const af::dtype& t) {
  switch (t) {
    case af::dtype::f16:
      return miopenHalf;
    case af::dtype::f32:
      return miopenFloat;
    case af::dtype::s32:
      return miopenInt32;
    default:
      throw std::invalid_argument(
          "unsupported data type for MiOpen type=" + std::to_string((int)t));
  }
}

miopenPoolingMode_t miopenMapToPoolingMode(const PoolingMode mode) {
  switch (mode) {
    case PoolingMode::MAX:
      return miopenPoolingMax;
    case PoolingMode::AVG_INCLUDE_PADDING:
      return miopenPoolingAverageInclusive;
    case PoolingMode::AVG_EXCLUDE_PADDING:
      return miopenPoolingAverage;
    default:
      throw std::invalid_argument("unsupported pooling mode for MiOpen");
  }
}

miopenRNNMode_t miopenMapToRNNMode(const RnnMode mode) {
  switch (mode) {
    case RnnMode::RELU:
      return miopenRNNRELU;
    case RnnMode::TANH:
      return miopenRNNTANH;
    case RnnMode::LSTM:
      return miopenLSTM;
    case RnnMode::GRU:
      return miopenGRU;
    default:
      throw std::invalid_argument("unsupported RNN mode for MiOpen");
  }
}

TensorDescriptor::TensorDescriptor(const Variable& input)
    : TensorDescriptor(input.array()) {}

TensorDescriptor::TensorDescriptor(
    const af::dtype type,
    const af::dim4& af_dims) {
  MIOPEN_CHECK_ERR(miopenCreateTensorDescriptor(&descriptor));
  miopenDataType_t miopentype = miopenMapToType(type);

  std::array<int, 4> dims = {
      (int)af_dims[3], (int)af_dims[2], (int)af_dims[1], (int)af_dims[0]};

  // Sets strides so array is contiguous row-major for miopen
  std::vector<int> r_strides = {1};
  for (auto it = dims.rbegin(); it != dims.rend() - 1; ++it) {
    r_strides.push_back(r_strides.back() * (*it));
  }
  std::vector<int> strides(r_strides.rbegin(), r_strides.rend());

  MIOPEN_CHECK_ERR(miopenSetTensorDescriptor(
      descriptor, miopentype, dims.size(), dims.data(), strides.data()));
}

TensorDescriptor::TensorDescriptor(const af::array& input) {
  descriptor = createAndSetTensorDescriptor(input);
}

TensorDescriptor::~TensorDescriptor() {
  MIOPEN_CHECK_ERR(miopenDestroyTensorDescriptor(descriptor));
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
  MIOPEN_CHECK_ERR(miopenCreatePoolingDescriptor(&descriptor));
  std::array<int, 2> window = {(int)wy, (int)wx};
  std::array<int, 2> padding = {(int)py, (int)px};
  std::array<int, 2> stride = {(int)sy, (int)sx};

  auto miopenPoolingMode = miopenMapToPoolingMode(mode);
  MIOPEN_CHECK_ERR(miopenSetNdPoolingDescriptor(
      descriptor,
      miopenPoolingMode,
      /* nbDims = */ 2,
      window.data(),
      padding.data(),
      stride.data()));
}

PoolingDescriptor::~PoolingDescriptor() {
  MIOPEN_CHECK_ERR(miopenDestroyPoolingDescriptor(descriptor));
}

FilterDescriptor::FilterDescriptor(const Variable& input)
    : FilterDescriptor(input.array()) {}

FilterDescriptor::FilterDescriptor(const af::array& input) {
  descriptor = createAndSetTensorDescriptor(input);
}

FilterDescriptor::~FilterDescriptor() {
  MIOPEN_CHECK_ERR(miopenDestroyTensorDescriptor(descriptor));
}

DropoutDescriptor::DropoutDescriptor(float drop_prob) {
  MIOPEN_CHECK_ERR(miopenCreateDropoutDescriptor(&descriptor));
  auto handle = getMiOpenHandle();
  unsigned long long seed = 0;
  size_t state_size;
  MIOPEN_CHECK_ERR(miopenDropoutGetStatesSize(handle, &state_size));
  auto& dropout_states = getDropoutStates();
  if (dropout_states.isempty()) {
    dropout_states = af::array(state_size, af::dtype::b8);
    DevicePtr statesraw(dropout_states);
    MIOPEN_CHECK_ERR(miopenSetDropoutDescriptor(
        descriptor,
        handle,
        drop_prob,
        statesraw.get(),
        state_size,
        seed,
        /* use_mask= */ true,
        /* state_evo= */ false,
        /* rng_mode= */ miopenRNGType_t::MIOPEN_RNG_PSEUDO_XORWOW));
  } else {
    DevicePtr statesraw(dropout_states);
    MIOPEN_CHECK_ERR(miopenRestoreDropoutDescriptor(
        descriptor,
        handle,
        drop_prob,
        statesraw.get(),
        state_size,
        seed,
        /* use_mask= */ true,
        /* state_evo= */ false,
        /* rng_mode= */ miopenRNGType_t::MIOPEN_RNG_PSEUDO_XORWOW));
  }
}

DropoutDescriptor::~DropoutDescriptor() {
  MIOPEN_CHECK_ERR(miopenDestroyDropoutDescriptor(descriptor));
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
  MIOPEN_CHECK_ERR(miopenCreateRNNDescriptor(&descriptor));

  auto handle = getMiOpenHandle();

  miopenRNNInputMode_t in_mode = miopenRNNlinear;

  miopenRNNDirectionMode_t dir =
      bidirectional ? miopenRNNbidirection : miopenRNNunidirection;

  miopenRNNMode_t cell = miopenMapToRNNMode(mode);
  miopenRNNAlgo_t algo = miopenRNNdefault;
  miopenDataType_t miopentype = miopenMapToType(type);

  MIOPEN_CHECK_ERR(miopenSetRNNDescriptor_V2(
      descriptor,
      hidden_size,
      num_layers,
      dropout.descriptor,
      in_mode,
      /* direction= */ dir,
      /* rnnMode= */ cell,
      /* biasMode= */ miopenRNNwithBias,
      algo,
      miopentype));
}

RNNDescriptor::~RNNDescriptor() {
  MIOPEN_CHECK_ERR(miopenDestroyRNNDescriptor(descriptor));
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
  MIOPEN_CHECK_ERR(miopenCreateConvolutionDescriptor(&descriptor));
  miopenDataType_t miopentype = miopenMapToType(type);
  std::array<int, 2> padding = {(int)py, (int)px};
  std::array<int, 2> stride = {(int)sy, (int)sx};
  std::array<int, 2> dilation = {(int)dy, (int)dx};

  MIOPEN_CHECK_ERR(miopenInitConvolutionNdDescriptor(
      descriptor,
      /* spatialDim= */ 2,
      padding.data(),
      stride.data(),
      dilation.data(),
      miopenConvolutionMode_t::miopenConvolution));

  if (groups > 1) {
    MIOPEN_CHECK_ERR(miopenSetConvolutionGroupCount(descriptor, groups));
  }
}

ConvDescriptor::~ConvDescriptor() {
  MIOPEN_CHECK_ERR(miopenDestroyConvolutionDescriptor(descriptor));
}

miopenHandle_t getMiOpenHandle() {
  int af_id = af::getDevice();
  return handles[af_id].handle;
}

const void* kOne(const af::dtype t) {
  switch (t) {
    case af::dtype::f16:
    case af::dtype::f32:
      return &kFloatOne;
    case af::dtype::f64:
    default:
      throw std::invalid_argument("unsupported data type for MiOpen");
  }
}

const void* kZero(const af::dtype t) {
  switch (t) {
    case af::dtype::f16:
    case af::dtype::f32:
      return &kFloatZero;
    case af::dtype::f64:
    default:
      throw std::invalid_argument("unsupported data type for MiOpen");
  }
}

} // namespace fl
