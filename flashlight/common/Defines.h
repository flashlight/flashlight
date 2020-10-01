/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdlib>

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

struct DistributedConstants {
  static constexpr const char* kMaxDevicePerNode = "MAX_DEVICE_PER_NODE";
  static constexpr const char* kFilePath = "FILE_PATH";
  static constexpr const std::size_t kCoalesceCacheSize =
      ((size_t)(20) << 20); // 20 MB
};

/** @} */

} // namespace fl
