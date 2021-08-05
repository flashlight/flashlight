/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// TODO:fl::Tensor {misc} remove me when not dependent on AF
namespace af {
class array;
}

namespace fl {

class Tensor;

/**
 * Block the calling host thread until all outstanding computation has
 * completed on all devices.
 *
 * The implementation of this function should synchronize any outstanding
 * computation abstractions, blocking accordingly.
 */
void sync();

/**
 * Block the calling host thread until all outstanding computation on the device
 * with the given ID has completed.
 *
 * @param[in] deviceId the id of the device on which to block until computation
 * has completed.
 */
void sync(int deviceId);

// TODO:fl::Tensor {signature}
/**
 * Launches computation, [usually] asynchronously, on operations needed to make
 * a particular value for a tensor available.
 *
 * There is no implementation requirement for this function (it can be a noop),
 * but it may be called by the user in areas with empirically high memory
 * pressure or where computation might be prudent to perform as per the user's
 * discretion.
 *
 * To block the calling thread until evaluation is complete, see `fl::sync`.
 *
 * @param[in] tensor the tensor on which to launch computation.
 */
void eval(af::array& array);
void eval(fl::Tensor& tensor);

/**
 * Returns the device ID of the active device in the current thread. This is
 * backend agnostic - the ID may correspond to a CUDA-device, an OpenCL device,
 * or other arbitrary hardware. The default device (in the case where operations
 * are occuring on the CPU) should give 0.
 *
 * If unimplemented, an implementation should return 0.
 *
 * @return the active device ID
 */
int getDevice();

/**
 * Sets the active device in the current thread. This is backend agnostic - the
 * ID may correspond to a CUDA-device, an OpenCL device, or other arbitrary
 * hardware. The default device is 0.
 *
 * @param[in] deviceId
 */
void setDevice(int deviceId);

} // namespace fl
