/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/tensor/backend/onednn/DnnlUtils.h"

#include <stdexcept>
#include <utility>

#if FL_BACKEND_OPENCL
  #include <dnnl_ocl.hpp>
#endif

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/TensorBase.h"

#if FL_BACKEND_OPENCL
  #include "flashlight/fl/common/OpenClUtils.h"
#endif

namespace fl::detail {

DnnlStream::DnnlStream(dnnl::engine engine) {
#if FL_BACKEND_OPENCL
  stream_ = dnnl::ocl_interop::make_stream(engine, fl::ocl::getQueue());
#else
  stream_ = dnnl::stream(engine);
#endif
}

dnnl::stream& DnnlStream::getStream() {
  return stream_;
}

DnnlStream& DnnlStream::getInstance() {
  static DnnlStream instance(DnnlEngine::getInstance().getEngine());
  return instance;
}

DnnlEngine::DnnlEngine() {
#if FL_BACKEND_OPENCL
  engine_ = dnnl::ocl_interop::make_engine(
      fl::ocl::getDeviceId(), fl::ocl::getContext());
#else
  engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
#endif
}

dnnl::engine& DnnlEngine::getEngine() {
  return engine_;
}

DnnlEngine& DnnlEngine::getInstance() {
  static DnnlEngine instance;
  return instance;
}

dnnl::memory::dims convertToDnnlDims(const std::vector<Dim>& shape) {
  return dnnl::memory::dims(shape.begin(), shape.end());
}

dnnl::memory::dims convertShapeToDnnlDims(const Shape& shape) {
  return convertToDnnlDims(shape.get());
}

DnnlMemoryWrapper::DnnlMemoryWrapper(
    const Tensor& tensor,
    dnnl::memory::dims dims,
    dnnl::memory::format_tag format) {
#if FL_BACKEND_OPENCL
  fl::ocl::DevicePtrOpenCl _devicePtr(tensor);
  cl_mem* buffer = _devicePtr.getAsClMem();
  devicePtr_ = std::move(_devicePtr);
#else
  devicePtr_ = fl::DevicePtr(tensor);
  void* buffer = devicePtr_.get();
#endif
  descriptor_ =
      dnnl::memory::desc({dims}, detail::dnnlMapToType(tensor.type()), format);
  memory_ = dnnl::memory(
      descriptor_, detail::DnnlEngine::getInstance().getEngine(), buffer);
}

DnnlMemoryWrapper& DnnlMemoryWrapper::operator=(DnnlMemoryWrapper&& other) {
  devicePtr_ = std::move(other.devicePtr_);
  memory_ = std::move(other.memory_);
  descriptor_ = std::move(other.descriptor_);
  return *this;
}

dnnl::memory DnnlMemoryWrapper::getMemory() const {
  return memory_;
}

dnnl::memory::desc DnnlMemoryWrapper::getDescriptor() const {
  return descriptor_;
}

dnnl::memory dnnlAlignOrdering(
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& netArgs,
    const dnnl::memory& memory,
    const dnnl::memory::desc& desc) {
  auto memoryOut = memory;
  if (memory.get_desc() != desc) {
    // use the ordering requested by the descriptor
    memoryOut =
        dnnl::memory(desc, detail::DnnlEngine::getInstance().getEngine());
    net.push_back(dnnl::reorder(memory, memoryOut));
    netArgs.push_back({{DNNL_ARG_FROM, memory}, {DNNL_ARG_TO, memoryOut}});
  }
  return memoryOut;
}

void executeNetwork(
    std::vector<dnnl::primitive>& net,
    std::vector<std::unordered_map<int, dnnl::memory>>& netArgs) {
  if (net.size() != netArgs.size()) {
    throw std::invalid_argument(
        "executeNetwork - given different size nets and netArgs");
  }
  // TODO{fl::Tensor}{macros} -- improve this to work with other backend interop
  // If on the CPU backend, there isn't a AF computation stream that facilitates
  // enforcing that inputs to computation are ready; we're required to wait
  // until all AF operations are done
  if (FL_BACKEND_CPU) {
    fl::sync();
  }

  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(DnnlStream::getInstance().getStream(), netArgs.at(i));
  }

  // TODO{fl::Tensor}{macros} -- improve this to work with other backend interop
  if (FL_BACKEND_CPU) {
    // Block the executing thread until the work is complete
    DnnlStream::getInstance().getStream().wait();
  }
}

dnnl::algorithm dnnlMapToPoolingMode(const PoolingMode mode) {
  switch (mode) {
    case PoolingMode::MAX:
      return dnnl::algorithm::pooling_max;
    case PoolingMode::AVG_INCLUDE_PADDING:
      return dnnl::algorithm::pooling_avg_include_padding;
    case PoolingMode::AVG_EXCLUDE_PADDING:
      return dnnl::algorithm::pooling_avg_exclude_padding;
    default:
      throw std::invalid_argument("unsupported pooling mode for cuDNN");
  }
}

} // namespace fl
