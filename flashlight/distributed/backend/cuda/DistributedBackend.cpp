/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

#include <mpi.h>
#include <nccl.h>

#include "flashlight/common/CudaUtils.h"
#include "flashlight/common/Defines.h"
#include "flashlight/common/DevicePtr.h"
#include "flashlight/distributed/DistributedApi.h"
#include "flashlight/distributed/FileStore.h"

#define NCCLCHECK(expr) ::fl::detail::ncclCheck((expr))
#define MPICHECK(expr) ::fl::detail::mpiCheck((expr))

namespace fl {

namespace detail {

namespace {

// We need to pass this flag to our dedicated NCCL CUDA stream, else activity in
// the stream will be precluded from running in parallel with the default stream
constexpr unsigned int kDefaultStreamFlags = cudaStreamNonBlocking;

constexpr const char* kNcclKey = "ncclUniqueId";

class NcclContext {
 public:
  static NcclContext& getInstance();
  NcclContext() = default;
  ~NcclContext();
  void initWithMPI(const std::unordered_map<std::string, std::string>& params);
  void initWithFileSystem(
      int worldRank,
      int worldSize,
      const std::unordered_map<std::string, std::string>& params);
  ncclComm_t& getComm();
  int getWorldSize() const;
  int getWorldRank() const;
  cudaStream_t getReductionStream() const;
  cudaStream_t getWorkerStream() const;
  cudaEvent_t getEvent() const;
  void* getCoalesceBuffer();

 private:
  // create CUDA resources
  void createCudaResources();
  ncclComm_t comm_;
  int worldSize_, worldRank_;
  // CUDA stream in which NCCL calls run if in async mode
  cudaStream_t reductionStream_;
  // CUDA stream in which cudaMemcpyAsync calls run if in contiguous mode
  cudaStream_t workerStream_;
  // Buffer for storing copied gradients contiguously; exists on device memory
  void* coalesceBuffer_{nullptr};
  std::once_flag allocBuffer_;
  // CUDA event to reuse for stream synchronization
  cudaEvent_t event_;
};

bool isNonNegativeInteger(const std::string& s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

ncclDataType_t getNcclTypeForArray(const af::array& arr) {
  switch (arr.type()) {
    case af::dtype::f32:
      return ncclFloat32;
    case af::dtype::f64:
      return ncclFloat64;
    case af::dtype::s32:
      return ncclInt32;
    case af::dtype::s64:
      return ncclInt64;
      break;
    default:
      throw std::runtime_error("unsupported data type for allreduce with NCCL");
  }
}

} // namespace

void ncclCheck(ncclResult_t r);

void mpiCheck(int ec);

void allreduceCuda(
    void* ptr,
    size_t count,
    ncclDataType_t ncclType,
    bool async,
    bool contiguous);
} // namespace detail

void allReduce(af::array& arr, bool async /* = false */) {
  if (!isDistributedInit()) {
    throw std::runtime_error("distributed environment not initialized");
  }
  ncclDataType_t type = detail::getNcclTypeForArray(arr);
  DevicePtr arrPtr(arr);
  detail::allreduceCuda(
      arrPtr.get(),
      arr.elements(),
      type,
      async,
      /* contiguous = */ false);
}

void allReduceMultiple(
    std::vector<af::array*> arrs,
    bool async /* = false */,
    bool contiguous /* = false */) {
  // Fast paths
  if (arrs.size() == 0) {
    return;
  }

  if (!contiguous) {
    // Use nccl groups to do everything in a single kernel launch
    NCCLCHECK(ncclGroupStart());
    for (auto& arr : arrs) {
      allReduce(*arr);
    }
    NCCLCHECK(ncclGroupEnd());
    return;
  }

  // We can only do a contiguous set reduction if all arrays in the set are of
  // the same type, else fail
  ncclDataType_t ncclType = detail::getNcclTypeForArray(*arrs[0]);
  for (auto& arr : arrs) {
    if (detail::getNcclTypeForArray(*arr) != ncclType) {
      throw std::runtime_error(
          "Cannot perform contiguous set allReduce on a set of tensors "
          "of different types");
    }
  }
  // Size of each element in each tensor in bytes
  size_t typeSize = af::getSizeOf(arrs[0]->type());

  // Device ptrs from each array
  std::vector<std::pair<DevicePtr, size_t>> arrPtrs;
  arrPtrs.reserve(arrs.size());
  size_t totalEls{0};
  for (auto& arr : arrs) {
    totalEls += arr->elements();
    arrPtrs.emplace_back(std::make_pair(DevicePtr(*arr), arr->bytes()));
  }

  // Make sure our coalesce buffer is large enough. Since we're initializing our
  // coalescing cache to the same size, if we're using contiguous sync, it
  // should never be larger since we flush if adding an additional buffer would
  // exceed the max cache size
  if (totalEls * typeSize > DistributedConstants::kCoalesceCacheSize) {
    throw std::runtime_error(
        "Total coalesce buffer size is larger than existing buffer size");
  }

  auto& ncclContext = detail::NcclContext::getInstance();
  cudaStream_t workerStream = ncclContext.getWorkerStream();
  // Block the copy worker stream on the ArrayFire CUDA stream
  cuda::synchronizeStreams(
      workerStream, cuda::getActiveStream(), ncclContext.getEvent());

  // In the worker stream, coalesce gradients into one large buffer so we
  // only need to call allReduce
  void* coalesceBuffer = ncclContext.getCoalesceBuffer();
  auto* cur = reinterpret_cast<char*>(coalesceBuffer);
  for (auto& entry : arrPtrs) {
    FL_CUDA_CHECK(cudaMemcpyAsync(
        cur,
        entry.first.get(),
        entry.second,
        cudaMemcpyDeviceToDevice,
        workerStream));
    cur += entry.second;
  }

  // Now, call allReduce once on the entire copy buffer
  detail::allreduceCuda(coalesceBuffer, totalEls, ncclType, async, contiguous);

  // Block the worker stream's copy operations on allReduce operations that are
  // currently enqueued in the reduction stream
  cudaStream_t syncStream;
  if (async) {
    syncStream = ncclContext.getReductionStream();
  } else {
    syncStream = cuda::getActiveStream();
  }
  cuda::synchronizeStreams(workerStream, syncStream, ncclContext.getEvent());

  // Enqueue operations in the stream to copy back to each respective array from
  // the coalesce buffer
  cur = reinterpret_cast<char*>(coalesceBuffer);
  for (auto& entry : arrPtrs) {
    FL_CUDA_CHECK(cudaMemcpyAsync(
        entry.first.get(),
        cur,
        entry.second,
        cudaMemcpyDeviceToDevice,
        workerStream));
    cur += entry.second;
  }
}

/**
 * Block future operations in the AF Stream on operations currently running in
 * the NCCL CUDA stream.
 */
void syncDistributed() {
  // If the worker or reduction streams have nothing enqueued, the AF CUDA
  // stream will proceed without waiting for anything
  auto& ncclContext = detail::NcclContext::getInstance();
  cuda::synchronizeStreams(
      cuda::getActiveStream(),
      ncclContext.getWorkerStream(),
      ncclContext.getEvent());
  cuda::synchronizeStreams(
      cuda::getActiveStream(),
      ncclContext.getReductionStream(),
      ncclContext.getEvent());
}

int getWorldRank() {
  if (!isDistributedInit()) {
    return 0;
  }
  return detail::NcclContext::getInstance().getWorldRank();
}

int getWorldSize() {
  if (!isDistributedInit()) {
    return 1;
  }
  return detail::NcclContext::getInstance().getWorldSize();
}

void distributedInit(
    DistributedInit initMethod,
    int worldRank,
    int worldSize,
    const std::unordered_map<std::string, std::string>& params /* = {} */) {
  if (isDistributedInit()) {
    std::cerr << "warning: fl::distributedInit() called more than once\n";
    return;
  }
  if (initMethod == DistributedInit::MPI) {
    detail::NcclContext::getInstance().initWithMPI(params);
    detail::DistributedInfo::getInstance().initMethod_ = DistributedInit::MPI;
  } else if (initMethod == DistributedInit::FILE_SYSTEM) {
    detail::NcclContext::getInstance().initWithFileSystem(
        worldRank, worldSize, params);
    detail::DistributedInfo::getInstance().initMethod_ =
        DistributedInit::FILE_SYSTEM;
  } else {
    throw std::runtime_error(
        "unsupported distributed init method for NCCL backend");
  }
  detail::DistributedInfo::getInstance().isInitialized_ = true;
  detail::DistributedInfo::getInstance().backend_ = DistributedBackend::NCCL;
  if (getWorldRank() == 0) {
    std::cout << "Initialized NCCL " << NCCL_MAJOR << "." << NCCL_MINOR << "."
              << NCCL_PATCH << " successfully!\n";
  }
}

namespace detail {

void ncclCheck(ncclResult_t r) {
  if (r == ncclSuccess) {
    return;
  }
  const char* err = ncclGetErrorString(r);
  if (r == ncclInvalidArgument) {
    throw std::invalid_argument(err);
  } else if (r == ncclInvalidUsage) {
    throw std::logic_error(err);
  } else {
    throw std::runtime_error(err);
  }
}

void mpiCheck(int ec) {
  if (ec == MPI_SUCCESS) {
    return;
  } else {
    char buf[MPI_MAX_ERROR_STRING];
    int resultlen;
    MPI_Error_string(ec, buf, &resultlen);
    throw std::runtime_error(buf);
  }
}

void allreduceCuda(
    void* ptr,
    size_t count,
    ncclDataType_t ncclType,
    bool async,
    bool contiguous) {
  cudaStream_t syncStream;
  auto& ncclContext = detail::NcclContext::getInstance();
  if (async) {
    syncStream = ncclContext.getReductionStream();
  } else {
    // AF CUDA stream
    syncStream = cuda::getActiveStream(); // assumes current device id
  }

  // Synchronize with whatever CUDA stream is performing operations needed
  // pre-reduction. If we're in contiguous mode, we need the reduction stream to
  // wait for the copy in the worker stream to complete. If we're not in
  // contiguous mode, we need to wait for the JIT eval triggered by acquisition
  // of the af::array's device pointer to complete, which will occur in the AF
  // CUDA stream.
  if (contiguous) {
    // block future reduction stream ops on the copy-worker stream
    cuda::synchronizeStreams(
        syncStream, ncclContext.getWorkerStream(), ncclContext.getEvent());
  } else {
    // block future reduction stream ops on the AF CUDA stream
    if (async) {
      cuda::synchronizeStreams(
          syncStream, cuda::getActiveStream(), ncclContext.getEvent());
    }
    // don't synchronize streams if not async and not contiguous - the AF CUDA
    // stream does everything
  }

  NCCLCHECK(ncclAllReduce(
      ptr, ptr, count, ncclType, ncclSum, ncclContext.getComm(), syncStream));
}
namespace {

ncclComm_t& NcclContext::getComm() {
  return comm_;
}

int NcclContext::getWorldSize() const {
  return worldSize_;
}

int NcclContext::getWorldRank() const {
  return worldRank_;
}

cudaStream_t NcclContext::getReductionStream() const {
  return reductionStream_;
}

cudaStream_t NcclContext::getWorkerStream() const {
  return workerStream_;
}

cudaEvent_t NcclContext::getEvent() const {
  return event_;
}

void* NcclContext::getCoalesceBuffer() {
  std::call_once(allocBuffer_, [&]() {
    FL_CUDA_CHECK(
        cudaMalloc(&coalesceBuffer_, DistributedConstants::kCoalesceCacheSize));
  });
  return coalesceBuffer_;
}

/* static */ NcclContext& NcclContext::getInstance() {
  static NcclContext ncclCtx;
  return ncclCtx;
}

void NcclContext::createCudaResources() {
  // initialize dedicated NCCL CUDA stream to support async allReduce
  FL_CUDA_CHECK(cudaStreamCreateWithFlags(
      &reductionStream_, detail::kDefaultStreamFlags));
  // initialize a third dedicated stream to asynchronously copy gradients
  // into a coalesced form if using a contiguous allReduce
  FL_CUDA_CHECK(
      cudaStreamCreateWithFlags(&workerStream_, detail::kDefaultStreamFlags));

  FL_CUDA_CHECK(cudaEventCreate(&event_, cuda::detail::kCudaEventDefaultFlags));
}

void NcclContext::initWithMPI(
    const std::unordered_map<std::string, std::string>& params) {
  // initializing MPI
  MPICHECK(MPI_Init(nullptr, nullptr));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank_));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize_));

  auto maxDevicePerNode = params.find(DistributedConstants::kMaxDevicePerNode);
  if (maxDevicePerNode == params.end() ||
      !isNonNegativeInteger(maxDevicePerNode->second) ||
      std::stoi(maxDevicePerNode->second) == 0) {
    throw std::invalid_argument(
        "invalid MaxDevicePerNode for NCCL initWithMPI");
  }

  ncclUniqueId id;

  // TODO: Determining device is ugly. Find a better way.
  af::setDevice(worldRank_ % std::stoi(maxDevicePerNode->second));

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (worldRank_ == 0) {
    ncclGetUniqueId(&id);
  }
  MPICHECK(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm_, worldSize_, id, worldRank_));

  createCudaResources();
}

void NcclContext::initWithFileSystem(
    int worldRank,
    int worldSize,
    const std::unordered_map<std::string, std::string>& params) {
  auto filePath = params.find(DistributedConstants::kFilePath);
  auto maxDevicePerNode = params.find(DistributedConstants::kMaxDevicePerNode);

  if (filePath == params.end() || filePath->second.empty()) {
    throw std::invalid_argument("invalid FilePath for NCCL initWithFileSystem");
  }
  if (maxDevicePerNode == params.end()) {
    throw std::invalid_argument(
        "invalid MaxDevicePerNode for NCCL initWithFileSystem");
  }

  worldRank_ = worldRank;
  worldSize_ = worldSize;

  ncclUniqueId id;

  af::setDevice(worldRank_ % std::stoi(maxDevicePerNode->second));

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (worldRank_ == 0) {
    ncclGetUniqueId(&id);
  }

  auto fs = FileStore(filePath->second);
  if (worldRank_ == 0) {
    std::vector<char> data(sizeof(id));
    std::memcpy(data.data(), &id, sizeof(id));
    fs.set(kNcclKey, data);
  } else {
    auto data = fs.get(kNcclKey);
    std::memcpy(&id, data.data(), sizeof(id));
  }
  // No need for barrier here as ncclCommInitRank inherently synchronizes

  // initializing NCCL
  NCCLCHECK(ncclCommInitRank(&comm_, worldSize_, id, worldRank_));

  createCudaResources();
}

NcclContext::~NcclContext() {
#ifdef NO_NCCL_COMM_DESTROY_HANDLE
// DEBUG : ncclCommDestroy disabled as it leads to segfault.
#else
  // finalizing NCCL
  NCCLCHECK(ncclCommDestroy(comm_));
#endif
#ifdef CUDA_NCCL_EVENT_DESTROY_ON_SHUTDOWN
  // destroy stream sync event
  FL_CUDA_CHECK(cudaEventDestroy(event_));
#endif

// Destroying the dedicated NCCL CUDA stream is a bit odd since the stream
// lives in a global NcclContext singleton. The CUDA driver shuts down when
// exit(0) is called, but the context may not be destroyed until
// afterwards, and destroying streams when the driver has already shut down is
// ungraceful. Manually destroying streams races against the driver, but in
// all cases, streams are destroyed when the driver shuts down, so don't
// destroy the stream by default.
#ifdef CUDA_STREAM_POOL_DESTROY_ON_SHUTDOWN
  FL_CUDA_CHECK(cudaStreamDestroy(reductionStream_));
  FL_CUDA_CHECK(cudaStreamDestroy(workerStream_));
#endif

// The CUDA driver has already shut down before we can free, so don't free by
// default, as driver shutdown will clean up this memory anyways.
#ifdef CUDA_CONTIGUOUS_BUFFER_FREE_ON_SHUTDOWN
  // Free the coalesce buffer if it was allocated
  if (coalesceBuffer_ != nullptr) {
    FL_CUDA_CHECK(cudaFree(coalesceBuffer_));
  }
#endif

  if (DistributedInfo::getInstance().initMethod_ == DistributedInit::MPI) {
    // finalizing MPI
    MPICHECK(MPI_Finalize());
  }
}
} // namespace
} // namespace detail

} // namespace fl
