/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/runtime/Device.h"
#include "flashlight/fl/runtime/DeviceType.h"
#include "flashlight/fl/runtime/Stream.h"

namespace fl {

class Tensor;

/**
 * Block the calling host thread until all outstanding computation has completed
 * on the active device with default type.
 *
 * The implementation of this function should synchronize any outstanding
 * computation abstractions, blocking accordingly.
 */
FL_API void sync();

/**
 * Block the calling host thread until all outstanding computation on the device
 * with the given ID and default type has completed.
 *
 * @param[in] deviceId the id of the device with default type on which to block
 * until computation has completed.
 */
FL_API void sync(const int deviceId);

/**
 * Block the calling host thread until all outstanding computation has completed
 * on all active devices for given types.
 *
 * @param[in] types the types of active devices to synchronize.
 */
FL_API void sync(const std::unordered_set<DeviceType>& types);

/**
 * Block the calling host thread until all outstanding computation has completed
 * on all given devices.
 *
 * @param[in] devices the devices to synchronize.
 */
FL_API void sync(const std::unordered_set<const Device*>& devices);

/**
 * Synchronize future tasks on given stream w.r.t. current tasks on all unique
 * streams of given tensors, i.e., the former can only start after the
 * completion of the latter.
 * NOTE this function may or may not block the calling thread.
 *
 * @param[in] wait the stream perform relative synchronization for.
 * @param[in] waitOns the tensors whose streams to perform relative
 * synchronization against.
 */
FL_API void relativeSync(
    const Stream& wait,
    const std::vector<const Tensor*>& waitOns);

/**
 * Synchronize future tasks on given stream w.r.t. current tasks on all unique
 * streams of given tensors, i.e., the former can only start after the
 * completion of the latter.
 * NOTE this function may or may not block the calling thread.
 *
 * @param[in] wait the stream perform relative synchronization for.
 * @param[in] waitOns the tensors whose streams to perform relative
 * synchronization against.
 */
FL_API void relativeSync(
    const Stream& wait,
    const std::vector<Tensor>& waitOns);

/**
 * Synchronize future tasks on the streams of `waits` w.r.t. current task on
 * given stream, i.e., the former can only start after the completion of the
 * latter. NOTE this function may or may not block the calling thread.
 *
 * @param[in] waits the tensors whose streams to perform relative
 * synchronization for.
 * @param[in] waitOn the stream to perform relative synchronization against.
 */
FL_API void relativeSync(
    const std::vector<Tensor>& waits,
    const Stream& waitOn);

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
FL_API void eval(fl::Tensor& tensor);

/**
 * Returns the device ID of the active device of default type in the current
 * thread. This is backend agnostic - the ID may correspond to a CUDA-device, an
 * OpenCL device, or other arbitrary hardware. The default device (in the case
 * where operations are occuring on the CPU) should give 0.
 *
 * If unimplemented, an implementation should return 0.
 *
 * TODO: eventually fold into Flashlight runtime
 *
 * @return the active device ID
 */
FL_API int getDevice();

/**
 * Sets the active device of default type in the current thread. This is backend
 * agnostic - the ID may correspond to a CUDA-device, an OpenCL device, or other
 * arbitrary hardware. The default device is 0.
 *
 * TODO: eventually fold into Flashlight runtime
 *
 * @param[in] deviceId
 */
FL_API void setDevice(const int deviceId);

/**
 * Gets the number of active devices.
 *
 * TODO: eventually fold into Flashlight runtime
 *
 * @returns the number of active devices usable in Flashlight.
 */
FL_API int getDeviceCount();

namespace detail {

/**
 * Write the current state of the memory manager to a specified output stream.
 * This function may be a noop for backends that do not implement memory
 * managers with configurable logging.
 */
FL_API void getMemMgrInfo(
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
FL_API void setMemMgrLogStream(std::ostream* stream);

/**
 * Sets (or unsets) logging for memory management. This function may be a noop
 * for backends that do not implement memory managers with configurable logging
 * capability.
 *
 * TODO: consolidate or improve this API
 *
 * @param[in] enabled true to enable logging, false to disable.
 */
FL_API void setMemMgrLoggingEnabled(const bool enabled);

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
FL_API void setMemMgrFlushInterval(const size_t interval);

} // namespace detail
} // namespace fl
