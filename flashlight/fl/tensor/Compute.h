/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <iostream>

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
void sync(const int deviceId);

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
 * TODO: eventually fold into Flashlight runtime
 *
 * @param[in] tensor the tensor on which to launch computation.
 */
void eval(fl::Tensor& tensor);

/**
 * Returns the device ID of the active device in the current thread. This is
 * backend agnostic - the ID may correspond to a CUDA-device, an OpenCL device,
 * or other arbitrary hardware. The default device (in the case where operations
 * are occuring on the CPU) should give 0.
 *
 * If unimplemented, an implementation should return 0.
 *
 * TODO: eventually fold into Flashlight runtime
 *
 * @return the active device ID
 */
int getDevice();

/**
 * Sets the active device in the current thread. This is backend agnostic - the
 * ID may correspond to a CUDA-device, an OpenCL device, or other arbitrary
 * hardware. The default device is 0.
 *
 * TODO: eventually fold into Flashlight runtime
 *
 * @param[in] deviceId
 */
void setDevice(const int deviceId);

/**
 * Gets the number of active devices.
 *
 * TODO: eventually fold into Flashlight runtime
 *
 * @returns the number of active devices usable in Flashlight.
 */
int getDeviceCount();

namespace detail {

/**
 * Write the current state of the memory manager to a specified output stream.
 * This function may be a noop for backends that do not implement memory
 * managers with configurable logging.
 */
void getMemMgrInfo(
    const char* msg,
    const int deviceId,
    std::ostream* ostream = &std::cout);

/**
 * Configures memory manager log output to write to a specified output stream.
 * This function may be a noop for backends that do not implement memory
 * managers with configurable logging.
 *
 * TODO: consolidate or improve this API
 *
 * @returns the number of active devices usable in Flashlight.
 */
void setMemMgrLogStream(std::ostream* stream);

/**
 * Sets (or unsets) logging for memory management. This function may be a noop
 * for backends that do not implement memory managers with configurable logging
 * capability.
 *
 * TODO: consolidate or improve this API
 *
 * @param[in] enabled true to enable logging, false to disable.
 */
void setMemMgrLoggingEnabled(const bool enabled);

/**
 * Configures memory manager log output to flush to the output stream after a
 * given number of lines are written. This function may be a noop
 * for backends that do not implement memory managers with configurable logging
 *
 * TODO: consolidate or improve this API
 *
 * @param[in] interval the number of lines after which to flush the temporary
 * log buffer. Supplied interval must be greater than 1.
 */
void setMemMgrFlushInterval(const size_t interval);

} // namespace detail
} // namespace fl
