/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/distributed/DistributedApi.h"

#include <iostream>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <gloo/allreduce_halving_doubling.h>
#include <gloo/config.h>
#include <gloo/mpi/context.h>
#include <gloo/transport/tcp/device.h>
#include <mpi.h>

#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/distributed/LRUCache.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace {
std::shared_ptr<gloo::mpi::Context> glooContext_;

// Gloo algorithms are "not meant" to be created an deleted often, for some
// strange reason. Therefore, we emulate THD by providing a cache of the last
// few algorithms run. See https://git.io/fNNyc
//
// XXX IMPORTANT XXX
// Caching is performed by raw pointer address. THIS MEANS YOU SHOULD NEVER PASS
// ANY TEMPORARIES TO THE ALLREDUCE/ALLGATHER/BRODACAST CONVENIENCE FUNCTIONS!
// If you do, you might eventually experience hangs and timeouts as temporaries
// are being allocated to previous memory addresses.
const int kGlooCacheSize_ = 10;
using CacheType = fl::detail::LRUCache<std::string, gloo::Algorithm>;
CacheType glooCache_(kGlooCacheSize_);
fl::Tensor cacheTensor_;
} // namespace

namespace fl {

namespace detail {

std::shared_ptr<gloo::mpi::Context> globalContext() {
  return glooContext_;
}

template <typename T>
inline void allreduceGloo(T* ptr, size_t s) {
  auto key = detail::makeHashKey(ptr, s, "allreduceCpu");
  auto algorithm = glooCache_.get(key);
  if (algorithm == nullptr) {
    using Allreduce = gloo::AllreduceHalvingDoubling<T>;
    algorithm = glooCache_.put(
        key,
        std::make_unique<Allreduce>(
            globalContext(),
            std::vector<T*>({ptr}),
            s,
            gloo::ReductionFunction<T>::sum));
  }
  algorithm->run();
}
} // namespace detail

void distributedInit(
    DistributedInit initMethod,
    int /* worldRank */,
    int /* worldSize */,
    const std::unordered_map<std::string, std::string>& /* params = {} */) {
  if (isDistributedInit()) {
    std::cerr << "warning: fl::distributedInit() called more than once\n";
    return;
  }

  if (initMethod != DistributedInit::MPI) {
    throw std::runtime_error(
        "unsupported distributed init method for gloo backend");
  }

  // using MPI
  if (glooContext_ != nullptr) {
    return;
  }
  // TODO: ibverbs support.
  auto glooDev = gloo::transport::tcp::CreateDevice("");

  // Create Gloo context from MPI communicator
  glooContext_ = gloo::mpi::Context::createManaged();
  glooContext_->setTimeout(gloo::kNoTimeout);
  glooContext_->connectFullMesh(glooDev);

  detail::DistributedInfo::getInstance().backend_ = DistributedBackend::GLOO;
  detail::DistributedInfo::getInstance().isInitialized_ = true;
  if (glooContext_->rank == 0) {
    std::cout << "Initialized Gloo successfully!\n";
  }
}

void allReduce(fl::Tensor& tensor, bool async /* = false */) {
  if (!isDistributedInit()) {
    throw std::runtime_error("distributed environment not initialized");
  }
  if (async) {
    throw std::runtime_error(
        "Asynchronous allReduce not yet supported for Gloo backend");
  }
  size_t tensorSize = tensor.elements() * fl::getTypeSize(tensor.type());
  if (tensorSize > cacheTensor_.elements()) {
    cacheTensor_ =
        fl::Tensor({static_cast<long long>(tensorSize)}, fl::dtype::b8);
  }
  DevicePtr tensorPtr(tensor);
  DevicePtr cacheTensorPtr(cacheTensor_);
  memcpy(cacheTensorPtr.get(), tensorPtr.get(), tensorSize);
  switch (tensor.type()) {
    case fl::dtype::f32:
      detail::allreduceGloo(
          static_cast<float*>(cacheTensorPtr.get()), tensor.elements());
      break;
    case fl::dtype::f64:
      detail::allreduceGloo(
          static_cast<double*>(cacheTensorPtr.get()), tensor.elements());
      break;
    case fl::dtype::s32:
      detail::allreduceGloo(
          static_cast<int*>(cacheTensorPtr.get()), tensor.elements());
      break;
    case fl::dtype::s64:
      detail::allreduceGloo(
          static_cast<int64_t*>(cacheTensorPtr.get()), tensor.elements());
      break;
    default:
      throw std::runtime_error("unsupported data type for allreduce with gloo");
  }
  memcpy(tensorPtr.get(), cacheTensorPtr.get(), tensorSize);
}

// Not yet supported
void allReduceMultiple(
    std::vector<fl::Tensor*> tensors,
    bool async /* = false */,
    bool contiguous /* = false */) {
  if (contiguous) {
    throw std::runtime_error(
        "contiguous allReduceMultiple is not yet supported for Gloo backend");
  }

  for (auto& tensor : tensors) {
    allReduce(*tensor, async);
  }
}

void syncDistributed() {
  // NOOP since async distributed operations aren't yet supported with the Gloo
  // backend
  return;
}

int getWorldRank() {
  if (!isDistributedInit()) {
    return 0;
  }
  return detail::globalContext()->rank;
}

int getWorldSize() {
  if (!isDistributedInit()) {
    return 1;
  }
  return detail::globalContext()->size;
}
} // namespace fl
