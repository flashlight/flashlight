/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MkldnnUtils.h"

namespace fl {
namespace detail {

mkldnn::stream& MkldnnStream::getStream() {
  return stream_;
}

MkldnnStream& MkldnnStream::getInstance() {
  static MkldnnStream instance;
  return instance;
}

mkldnn::engine& MkldnnEngine::getEngine() {
  return engine_;
}

MkldnnEngine& MkldnnEngine::getInstance() {
  static MkldnnEngine instance;
  return instance;
}

mkldnn::memory::dims convertAfToMklDnnDims(const std::vector<dim_t>& afDims) {
  // MKL-DNN uses ints in dims
  std::vector<int> intVec(afDims.begin(), afDims.end());
  return mkldnn::memory::dims(intVec);
}

mkldnn::memory mkldnnAlignOrdering(
    std::vector<mkldnn::primitive>& net,
    const mkldnn::memory& memory,
    const mkldnn::memory::primitive_desc& desc) {
  auto memoryOut = memory;
  if (memory.get_primitive_desc() != mkldnn::memory::primitive_desc(desc)) {
    memoryOut =
        mkldnn::memory(desc); // use the ordering requested by the descriptor
    net.push_back(mkldnn::reorder(memory, memoryOut));
  }
  return memoryOut;
}

} // namespace detail
} // namespace fl
