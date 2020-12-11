/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>

#include <arrayfire.h>
#include <dnnl.hpp>

#include "flashlight/fl/common/Defines.h"

namespace fl {
namespace detail {

/**
 * A singleton class that contains a static instance of a dnnl::stream.
 */
class DnnlStream {
 public:
  DnnlStream(dnnl::engine engine) : stream_(engine) {}
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
  DnnlEngine() : engine_(dnnl::engine::kind::cpu, 0) {}
  ~DnnlEngine() = default;

  /// Prohibit assignment
  DnnlEngine& operator=(DnnlEngine const& e) = delete;

  dnnl::engine& getEngine();

  static DnnlEngine& getInstance();

 private:
  dnnl::engine engine_;
};

/**
 * Helper for converting an ArrayFire af::dim4 into an DNNL-compatible input
 * for dnnl::memory::dims.
 */
dnnl::memory::dims convertAfToDnnlDims(const std::vector<dim_t>& dims);

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
 * For each primitive, passes the corresponding arguments map for that index to
 * the execution stream. The number of primitives and the number of arguments
 * must be equal, else throws.
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
inline dnnl::memory::data_type dnnlMapToType(const af::dtype t) {
  if (t == af::dtype::f16) {
    return dnnl::memory::data_type::f16;
  } else if (t == af::dtype::f32) {
    return dnnl::memory::data_type::f32;
  } else if (t == af::dtype::f64) {
    throw std::invalid_argument("float64 is not supported by DNNL");
  } else {
    throw std::invalid_argument("data type not supported with DNNL");
  }
}

} // namespace detail
} // namespace fl
