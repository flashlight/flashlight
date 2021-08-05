/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
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

void sync(int deviceId) {
  Tensor().backend().sync(deviceId);
}

void eval(Tensor& tensor) {
  Tensor().backend().eval(tensor);
}

// TODO:fl::Tensor remove once no more `fl::eval` calls that take `af::array`s
void eval(af::array& array) {
  af::eval(array);
}

int getDevice() {
  return Tensor().backend().getDevice();
}

void setDevice(int deviceId) {
  Tensor().backend().setDevice(deviceId);
}

} // namespace fl
