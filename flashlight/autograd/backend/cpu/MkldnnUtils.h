/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <array>

#include <arrayfire.h>
#include <mkldnn.hpp>

#include "flashlight/flashlight/common/Defines.h"

namespace fl {
namespace detail {

/**
 * A singleton class that contains a static instance of a mkldnn::stream.
 */
class MkldnnStream {
 public:
  MkldnnStream() : stream_(mkldnn::stream::kind::eager) {}
  ~MkldnnStream() = default;

  /// Prohibit assignment
  MkldnnStream& operator=(MkldnnStream const& s) = delete;

  mkldnn::stream& getStream();

  static MkldnnStream& getInstance();

 private:
  mkldnn::stream stream_;
};

/**
 * A singleton class that contains a static instance of a mkldnn::engine.
 */
class MkldnnEngine {
 public:
  MkldnnEngine() : engine_(mkldnn::engine::cpu, 0) {}
  ~MkldnnEngine() = default;

  /// Prohibit assignment
  MkldnnEngine& operator=(MkldnnEngine const& e) = delete;

  mkldnn::engine& getEngine();

  static MkldnnEngine& getInstance();

 private:
  mkldnn::engine engine_;
};

/**
 * Helper for converting an ArrayFire af::dim4 into an MKL-DNN-compatible input
 * for mkldnn::memory::dims.
 */
mkldnn::memory::dims convertAfToMklDnnDims(const std::vector<dim_t>& dims);

/**
 * Given some an mkldnn network (a ``std::vector<mkldnn::primitive>``), a
 * ``mkldnn::memory`` with some ordering, and a
 * ``mkldnn::memory::primitive_desc``, determines whether or not the memory
 * needs to be ordered based on the primitive descriptor's required ordering.
 *
 * If so, adds a ``mkldnn::reorder`` layer to the network, and returns a new
 * memory descriptor that will be properly reordered.
 */
mkldnn::memory mkldnnAlignOrdering(
    std::vector<mkldnn::primitive>& net,
    const mkldnn::memory& memory,
    const mkldnn::memory::primitive_desc& desc);

/**
 * Given a flashlight pooling mode, returns the corresponding mkldnn pooling
 * mode.
 */
mkldnn::algorithm mkldnnMapToPoolingMode(const PoolingMode mode);

/**
 * Maps an ArrayFire array datatype into the corresponding MKL-DNN datatype.
 *
 * Needs to be explicitly inlined due to a bug with MKL-DNN.
 */
inline mkldnn::memory::data_type mkldnnMapToType(const af::dtype t) {
  if (t == af::dtype::f32) {
    return mkldnn::memory::data_type::f32;
  } else if (t == af::dtype::f64) {
    throw std::invalid_argument("float64 is not supported by MKL-DNN");
  } else {
    throw std::invalid_argument("data type not supported with MKL-DNN");
  }
}

} // namespace detail
} // namespace fl
