/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/af/ArrayFireBackend.h"

#include <af/algorithm.h>
#include <af/backend.h>
#include <af/data.h>
#include <af/device.h>
#include <af/random.h>

#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/backend/af/ArrayFireTensor.h"
#include "flashlight/fl/tensor/backend/af/Utils.h"
#include "flashlight/fl/tensor/backend/af/mem/MemoryManagerInstaller.h"

#if FL_ARRAYFIRE_USE_CPU
  #include "flashlight/fl/tensor/backend/af/ArrayFireCPUStream.h"
#endif

#if FL_ARRAYFIRE_USE_CUDA
  #include <cublas_v2.h>
  #include <cuda.h>
  #include <cuda_runtime.h>

  #include <af/cuda.h>

  #include "flashlight/fl/runtime/CUDAStream.h"
#endif

namespace fl {

namespace {

// Get the stream associated with given device in the given map; if it's not in
// the map, initialize it (by wrapping or creating) and put it into the map.
const Stream& getOrWrapAfDeviceStream(
    const int afId,
    const int nativeId,
    std::unordered_map<int, std::shared_ptr<const Stream>>& afIdToStream) {
  auto iter = afIdToStream.find(afId);
  if (iter != afIdToStream.end()) {
    return *iter->second;
  }

#if FL_ARRAYFIRE_USE_CPU
  auto resIter = afIdToStream.emplace(afId, ArrayFireCPUStream::create());
  return *resIter.first->second;
#elif FL_ARRAYFIRE_USE_CUDA
  const cudaStream_t cudaNativeStream = afcu::getStream(afId);
  auto resIter = afIdToStream.emplace(
      afId, CUDAStream::wrapUnmanaged(nativeId, cudaNativeStream));
  return *resIter.first->second;
#else
  throw std::runtime_error(
      "ArrayFireBackend was not compiled with support for CPU or GPU");
#endif
}

} // namespace

ArrayFireBackend::ArrayFireBackend() {
  AF_CHECK(af_init());

  std::call_once(memoryInitFlag, []() {
    // TODO: remove this temporary workaround for TextDatasetTest crash on CPU
    // backend when tearing down the test environment. This is possibly due to
    // AF race conditions when tearing down our custom memory manager.
    // TODO: remove this temporary workaround for crashes when using custom
    // opencl kernels.
    if (FL_BACKEND_CUDA) {
      MemoryManagerInstaller::installDefaultMemoryManager();
    }
  });

  for (int id = 0; id < af::getDeviceCount(); id++) {
    int nativeId = id; // TODO investigate how OpenCL fits into this.
#if FL_ARRAYFIRE_USE_CUDA
    nativeId = afcu::getNativeId(id);
#endif
    // TODO make these maps `const`
    nativeIdToId_[nativeId] = id;
    idToNativeId_[id] = nativeId;
  }

  const auto& manager = DeviceManager::getInstance();
  // This callback ensures consistency of AF internal state on active device.
  // Capturing by value to avoid destructor race hazard for static objects.
  const auto setActiveCallback = [nativeIdToId = nativeIdToId_,
                                  afIdToStream = afIdToStream_](int nativeId) {
    auto afId = nativeIdToId.at(nativeId);
    af::setDevice(afId);
    // this is the latest point we can lazily wrap the AF stream, which may get
    // lazily intialized anytime in AF internally, e.g., via tensor computation.
    getOrWrapAfDeviceStream(afId, nativeId, *afIdToStream);
  };
#if FL_ARRAYFIRE_USE_CPU
  auto& device = manager.getActiveDevice(DeviceType::x64);
  device.addSetActiveCallback(setActiveCallback);
#elif FL_ARRAYFIRE_USE_CUDA
  const auto deviceCount = manager.getDeviceCount(DeviceType::CUDA);
  for (unsigned nativeId = 0; nativeId < deviceCount; nativeId++) {
    auto& device = manager.getDevice(DeviceType::CUDA, nativeId);
    device.addSetActiveCallback(setActiveCallback);
  }
#endif
  // Active device is never set explicitly, so we must wrap its stream eagerly.
  auto activeAfId = af::getDevice();
  getOrWrapAfDeviceStream(
      activeAfId, idToNativeId_.at(activeAfId), *afIdToStream_);
}

ArrayFireBackend& ArrayFireBackend::getInstance() {
  static ArrayFireBackend instance;
  return instance;
}

TensorBackendType ArrayFireBackend::backendType() const {
  return TensorBackendType::ArrayFire;
}

/* -------------------------- Compute Functions -------------------------- */

void ArrayFireBackend::eval(const Tensor& tensor) {
  af::eval(toArray(tensor));
}

const Stream& ArrayFireBackend::getStreamOfArray(
  const af::array& arr) {
  // TODO once we enforce integrate Device::setDevice into fl::setDevice, each
  // array's stream should always be wrapped already (via setDevice callback).
  // auto iter = afIdToStream_->find(af::getDeviceId(arr));
  // assert(iter != afIdToStream_->end() && "Stream should have been wrapped");
  // return *iter->second;
  auto afId = af::getDeviceId(arr);
  auto nativeId = idToNativeId_.at(afId);
  return getOrWrapAfDeviceStream(afId, nativeId, *afIdToStream_);
}

bool ArrayFireBackend::supportsDataType(const fl::dtype& dtype) const {
  switch (dtype) {
    case fl::dtype::f16:
      return af::isHalfAvailable(af::getDevice()) &&
          // f16 isn't [yet] supported with the CPU backend per onednn
          // limitations
          !FL_BACKEND_CPU;
    default:
      return true;
  }
}

void ArrayFireBackend::getMemMgrInfo(
    const char* msg,
    const int nativeDeviceId,
    std::ostream* ostream) {
  int deviceId = nativeIdToId_.at(nativeDeviceId);
  if (ostream == nullptr) {
    throw std::invalid_argument(
        "ArrayFireBackend::getMemMgrInfo - got null ostream pointer");
  }
  auto* curMemMgr =
      fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
  if (curMemMgr) {
    curMemMgr->printInfo(msg, deviceId, ostream);
  }
}

void ArrayFireBackend::setMemMgrLogStream(std::ostream* stream) {
  if (stream == nullptr) {
    throw std::invalid_argument(
        "ArrayFireBackend::getMemMgrInfo - got null ostream pointer");
  }
  auto* curMemMgr =
      fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
  if (curMemMgr) {
    curMemMgr->setLogStream(stream);
  }
}

void ArrayFireBackend::setMemMgrLoggingEnabled(const bool enabled) {
  auto* curMemMgr =
      fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
  if (curMemMgr) {
    curMemMgr->setLoggingEnabled(enabled);
  }
}

void ArrayFireBackend::setMemMgrFlushInterval(const size_t interval) {
  auto* curMemMgr =
      fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
  if (curMemMgr) {
    curMemMgr->setLogFlushInterval(interval);
  }
}

/* -------------------------- Rand Functions -------------------------- */

void ArrayFireBackend::setSeed(const int seed) {
  af::setSeed(seed);
}

Tensor ArrayFireBackend::randn(const Shape& shape, dtype type) {
  return toTensor<ArrayFireTensor>(
      af::randn(detail::flToAfDims(shape), detail::flToAfType(type)),
      shape.ndim());
}

Tensor ArrayFireBackend::rand(const Shape& shape, dtype type) {
  return toTensor<ArrayFireTensor>(
      af::randu(detail::flToAfDims(shape), detail::flToAfType(type)),
      shape.ndim());
}

/* --------------------------- Tensor Operators --------------------------- */

/******************** Tensor Creation Functions ********************/
#define AF_BACKEND_CREATE_FUN_LITERAL_DEF(TYPE)                          \
  Tensor ArrayFireBackend::fromScalar(TYPE value, const dtype type) {    \
    return toTensor<ArrayFireTensor>(                                    \
        af::constant(value, af::dim4(1), detail::flToAfType(type)),      \
        /* ndim = */ 0);                                                 \
  }                                                                      \
  Tensor ArrayFireBackend::full(                                         \
      const Shape& shape, TYPE value, const dtype type) {                \
    return toTensor<ArrayFireTensor>(                                    \
        af::constant(                                                    \
            value, detail::flToAfDims(shape), detail::flToAfType(type)), \
        shape.ndim());                                                   \
  }
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const double&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const float&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const int&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const char&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned char&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const long long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned long long&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const bool&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const short&);
AF_BACKEND_CREATE_FUN_LITERAL_DEF(const unsigned short&);

Tensor ArrayFireBackend::identity(const Dim dim, const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::identity({dim, dim}, detail::flToAfType(type)), /* numDims = */ 2);
}

Tensor ArrayFireBackend::arange(
    const Shape& shape,
    const Dim seqDim,
    const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::range(detail::flToAfDims(shape), seqDim, detail::flToAfType(type)),
      shape.ndim());
}

Tensor ArrayFireBackend::iota(
    const Shape& dims,
    const Shape& tileDims,
    const dtype type) {
  return toTensor<ArrayFireTensor>(
      af::iota(
          detail::flToAfDims(dims),
          detail::flToAfDims(tileDims),
          detail::flToAfType(type)),
      /* numDims = */ std::max(dims.ndim(), tileDims.ndim()));
}

Tensor ArrayFireBackend::where(
    const Tensor& condition,
    const Tensor& x,
    const Tensor& y) {
  Tensor orig = x;
  af::replace(toArray(orig), toArray(condition), toArray(y));
  return orig;
}

void ArrayFireBackend::topk(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const unsigned k,
    const Dim axis,
    const SortMode sortMode) {
  if (axis != 0) {
    throw std::invalid_argument(
        "ArrayFireTensor topk: operation only supported along zero axis.");
  }
  af::array valuesArr, indicesArr;
  af::topk(
      valuesArr,
      indicesArr,
      toArray(input),
      k,
      axis,
      detail::flToAfTopKSortMode(sortMode));

  values = toTensor<ArrayFireTensor>(std::move(valuesArr), input.ndim());
  indices = toTensor<ArrayFireTensor>(std::move(indicesArr), input.ndim());
}

Tensor ArrayFireBackend::sort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  if (sortMode != SortMode::Descending && sortMode != SortMode::Ascending) {
    throw std::invalid_argument(
        "Cannot sort ArrayFire tensor with given SortMode: "
        "only Descending and Ascending supported.");
  }

  af::array values, indices;
  af::sort(
      values, indices, toArray(input), axis, sortMode == SortMode::Ascending);
  return toTensor<ArrayFireTensor>(std::move(values), input.ndim());
}

void ArrayFireBackend::sort(
    Tensor& values,
    Tensor& indices,
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  if (sortMode != SortMode::Descending && sortMode != SortMode::Ascending) {
    throw std::invalid_argument(
        "Cannot sort ArrayFire tensor with given SortMode: "
        "only Descending and Ascending supported.");
  }

  af::array _values, _indices;
  af::sort(
      _values, _indices, toArray(input), axis, sortMode == SortMode::Ascending);
  values = toTensor<ArrayFireTensor>(std::move(_values), input.ndim());
  indices = toTensor<ArrayFireTensor>(std::move(_indices), input.ndim());
}

Tensor ArrayFireBackend::argsort(
    const Tensor& input,
    const Dim axis,
    const SortMode sortMode) {
  if (sortMode != SortMode::Descending && sortMode != SortMode::Ascending) {
    throw std::invalid_argument(
        "Cannot sort ArrayFire tensor with given SortMode: "
        "only Descending and Ascending supported.");
  }

  af::array values, indices;
  af::sort(
      values, indices, toArray(input), axis, sortMode == SortMode::Ascending);
  return toTensor<ArrayFireTensor>(std::move(indices), input.ndim());
}

void ArrayFireBackend::print(const Tensor& tensor) {
  af::print("ArrayFireTensor", toArray(tensor));
}
} // namespace fl
