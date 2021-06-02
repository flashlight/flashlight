/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/arith.h>
#include <af/device.h>
#include <af/exception.h>

#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"

/*
 * TODO: this is duplicative - remove this from flashlight/fl/common/Utils.h
 * once the rest of the proj depends on headers here.
 */
#define AF_CHECK(fn)                                                          \
  do {                                                                        \
    af_err __err = fn;                                                        \
    if (__err == AF_SUCCESS) {                                                \
      break;                                                                  \
    }                                                                         \
    throw af::exception(                                                      \
        "ArrayFire error: ", __PRETTY_FUNCTION__, __FILE__, __LINE__, __err); \
  } while (0)

namespace fl {

ArrayFireBackend::ArrayFireBackend() {
  AF_CHECK(af_init());
}

ArrayFireBackend& ArrayFireBackend::getInstance() {
  static ArrayFireBackend instance;
  return instance;
}

Tensor ArrayFireBackend::exp(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::exp(toArray(tensor)));
}

Tensor ArrayFireBackend::log(const Tensor& tensor) {
  return toTensor<ArrayFireTensor>(af::log(toArray(tensor)));
}

} // namespace fl
