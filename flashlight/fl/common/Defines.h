/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>
#include <string>
#include <unordered_map>

namespace fl {

/**
 * \defgroup common_defines Common constants and definitions
 * @{
 */

/**
 * Reduction mode to used for CrossEntropy, AdaptiveSoftMax etc ...
 */
enum class ReduceMode {
  NONE = 0,
  MEAN = 1,
  SUM = 2,
};

/**
 * Pooling method to be used
 */
enum class PoolingMode {

  /// Use maximum value inside the pooling window
  MAX = 0,

  /// Use average value (including padding) inside the pooling window
  AVG_INCLUDE_PADDING = 1,

  /// Use average value (excluding padding) inside the pooling window// Use
  /// average value (excluding padding) inside the pooling window
  AVG_EXCLUDE_PADDING = 2,
};

/**
 * RNN network type
 */
enum class RnnMode {
  RELU = 0,
  TANH = 1,
  LSTM = 2,
  GRU = 3,
};

enum class PaddingMode {
  /// Use smallest possible padding such that out_size = ceil(in_size/stride)
  SAME = -1,
};

enum class DistributedBackend {
  /// https://github.com/facebookincubator/gloo
  GLOO = 0,
  /// https://developer.nvidia.com/nccl
  NCCL = 1,
};

enum class DistributedInit {
  MPI = 0,
  FILE_SYSTEM = 1,
};

namespace DistributedConstants {
  constexpr const char* kMaxDevicePerNode = "MAX_DEVICE_PER_NODE";
  constexpr const char* kFilePath = "FILE_PATH";
  constexpr const std::size_t kCoalesceCacheSize =
      ((size_t)(20) << 20); // 20 MB
}

constexpr std::size_t kDynamicBenchmarkDefaultCount = 10;
constexpr double kAmpMinimumScaleFactorValue = 1e-4;

/**************************** Optimization Modes *****************************/
// TODO(jacobkahn): should we move this to a different header? In Types.h/cpp?

/**
 * Optimization levels in flashlight. These determine the computation behavior
 * of autograd operator computation as well as how inputs and outputs of
 * operators are cast.
 *
 * Operator precision roughly follows those found in NVIDIA Apex:
 * - https://bit.ly/33UpSWp
 * - https://bit.ly/30Zv2OS
 * - https://bit.ly/310k8Z6
 */
enum class OptimLevel {
  /// All operations occur in default (f32 or f64) precision.
  DEFAULT = 0,
  /// Operations that perform reduction accumulation, including layer/batch
  /// normalization are performed in f32 - all other operations are in fp16.
  /// To be used in a standard mixed-precision training setup.
  O1 = 1,
  /// Only batch and layer normalization occur in f32 - all other operations
  /// occur in f16.
  O2 = 2,
  /// All operations that support it use fp16.
  O3 = 3
};

/**
 * Singleton storing the current optimization level (`OptimLevel`) for
 * flashlight.
 */
class OptimMode {
 public:
  /**
   * @return the OptimMode singleton
   */
  static OptimMode& get();

  /**
   * Gets the current optimization level. Not thread safe.
   *
   * @return the current optimization level.
   */
  OptimLevel getOptimLevel();

  /**
   * Gets the current optimization level. Not thread safe.
   *
   * @param[in] level the optimization level to set
   */
  void setOptimLevel(OptimLevel level);

  /**
   *
   */
  static OptimLevel toOptimLevel(const std::string& in);

  static const std::unordered_map<std::string, OptimLevel> kStringToOptimLevel;

 private:
  OptimLevel optimLevel_{OptimLevel::DEFAULT};
};

/** @} */

} // namespace fl
