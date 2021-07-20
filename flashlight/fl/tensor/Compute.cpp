/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/Compute.h"

#include <utility>

#include "flashlight/fl/tensor/TensorBackend.h"
#include "flashlight/fl/tensor/TensorBase.h"

// TODO: remove me once no more `af::eval` calls
#include <af/array.h>
#include <af/device.h>
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

namespace fl {

void sync() {
  Tensor().backend().sync();
}

void eval(Tensor& tensor) {
  Tensor().backend().eval(tensor);
}

// TODO:fl::Tensor remove once no more `af::eval` calls
void eval(af::array& array) {
  af::array a = array;
  Tensor t = toTensor<ArrayFireTensor>(std::move(a));
  fl::eval(t);
}

int getDevice() {
  return Tensor().backend().getDevice();
}

void setDevice(int deviceId) {
  Tensor().backend().setDevice(deviceId);
}

} // namespace fl
