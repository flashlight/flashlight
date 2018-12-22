/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>

#include <flashlight/autograd/Variable.h>
#include <flashlight/common/Defines.h>

namespace fl {

/**
 * Initialize the distributed environment. Note that `worldSize`, `worldRank`
 * are ignored if DistributedInit::MPI is used.
 *
 * @param initMethod Initialization method used for setting up the rendezvous
 * @param worldSize Total number of processes in the communication group
 *`@param worldRank 0-indexed rank of the current process
 * @param params Additional parameters (if any) needed for initialization
 */
void distributedInit(
    DistributedInit initMethod,
    int worldRank,
    int worldSize,
    const std::unordered_map<std::string, std::string>& params = {});

/**
 * Returns whether the distributed environment has been initialized
 */
bool isDistributedInit();

/**
 * Returns the backend used for distributed setup
 */
DistributedBackend distributedBackend();

/**
 * Returns rank of the current process in the communication group (zero-based).
 * Returns 0 if distributed environment is not initialized
 */
int getWorldRank();

/**
 * Returns total process in the communication group
 * Returns 1 if distributed environment is not initialized
 */
int getWorldSize();

/**
 * Synchronizes a the array wrapped by the Variable with allreduce.
 *
 * @param var a variable whose array will be synchronized
 * @param scale scale the Variable after allreduce by this factor
 */
void allReduce(Variable& var, double scale = 1.0);

/**
 * Synchronizes a single Arrayfire array with allreduce.
 *
 * @param arr an array which will be synchronized
 */
void allReduce(af::array& arr);

namespace detail {
class DistributedInfo {
 public:
  static DistributedInfo& getInstance();

  bool isInitialized_ = false;
  DistributedInit initMethod_;
  DistributedBackend backend_;

 private:
  DistributedInfo() = default;
};
} // namespace detail

} // namespace fl
