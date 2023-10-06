/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>

#include <dnnl.hpp>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

class Tensor;

namespace detail {

/**
 * A singleton class that contains a static instance of a dnnl::stream.
 */
class DnnlStream {
 public:
  DnnlStream(dnnl::engine engine);
  ~DnnlStream() = default;

  /// Prohibit assignment
  DnnlStream& operator=(DnnlStream const& s) = delete;

  dnnl::stream& getStream();

  static DnnlStream& getInstance();

 private:
  dnnl::stream stream_;
};

/**
 * A singleton class that contains a static instance of a dnnl::engine.
 */
class DnnlEngine {
 public:
  DnnlEngine();
  ~DnnlEngine() = default;

  /// Prohibit assignment
  DnnlEngine& operator=(DnnlEngine const& e) = delete;

  dnnl::engine& getEngine();

  static DnnlEngine& getInstance();

 private:
  dnnl::engine engine_;
};

/**
 * Helper for converting a Flashlight Shape into an DNNL-compatible input
 * for dnnl::memory::dims.
 */
dnnl::memory::dims convertToDnnlDims(const std::vector<Dim>& dims);
dnnl::memory::dims convertShapeToDnnlDims(const Shape& shape);

/**
 * A light wrapper around dnnl::memory that manages underlying memory lifetime
 * in accordance with fl::DevicePtr.
 */
class DnnlMemoryWrapper {
 public:
  DnnlMemoryWrapper(
      const Tensor& tensor,
      dnnl::memory::dims dims,
      dnnl::memory::format_tag format);
  DnnlMemoryWrapper() = default;

  DnnlMemoryWrapper& operator=(DnnlMemoryWrapper&& other);

  dnnl::memory getMemory() const;

  dnnl::memory::desc getDescriptor() const;

 private:
  dnnl::memory::desc descriptor_;
  dnnl::memory memory_;
  fl::DevicePtr devicePtr_;
};

/**
 * Given some an dnnl network (a ``std::vector<dnnl::primitive>``), a
 * ``dnnl::memory`` with some ordering, and a
 * ``dnnl::memory::primitive_desc``, determines whether or not the memory
 * needs to be ordered based on the primitive descriptor's required ordering.
 *
 * If so, adds a ``dnnl::reorder`` layer to the network, and returns a new
 * memory descriptor that will be properly reordered.
 */
dnnl::memory dnnlAlignOrdering(
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& netArgs,
    const dnnl::memory& memory,
    const dnnl::memory::desc& desc);

/**
 * Executes a sequence of DNNL primitives in the default execution stream with
 * the default execution engine.
 *
 * For each primitive, passes the corresponding arguments map for that index
 * to the execution stream. The number of primitives and the number of
 * arguments must be equal, else throws.
 *
 * Blocks calling thread until the enqueued work has been completed.
 */
void executeNetwork(
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& args);

/**
 * Given a flashlight pooling mode, returns the corresponding dnnl pooling
 * mode.
 */
dnnl::algorithm dnnlMapToPoolingMode(const PoolingMode mode);

/**
 * Maps an ArrayFire array datatype into the corresponding DNNL datatype.
 *
 * Needs to be explicitly inlined due to a bug with DNNL.
 */
inline dnnl::memory::data_type dnnlMapToType(const fl::dtype t) {
  if (t == fl::dtype::f16) {
    return dnnl::memory::data_type::f16;
  } else if (t == fl::dtype::f32) {
    return dnnl::memory::data_type::f32;
  } else if (t == fl::dtype::f64) {
    throw std::invalid_argument("float64 is not supported by DNNL");
  } else {
    throw std::invalid_argument("data type not supported with DNNL");
  }
}

} // namespace detail
} // namespace fl
