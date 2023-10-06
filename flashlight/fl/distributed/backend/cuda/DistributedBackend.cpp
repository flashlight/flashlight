/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>

#include <mpi.h>
#include <nccl.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/DevicePtr.h"
#include "flashlight/fl/distributed/DistributedApi.h"
#include "flashlight/fl/distributed/FileStore.h"
#include "flashlight/fl/runtime/CUDAStream.h"
#include "flashlight/fl/runtime/CUDAUtils.h"
#include "flashlight/fl/runtime/DeviceManager.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Types.h"

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
  const CUDAStream& getReductionStream() const;
  const CUDAStream& getWorkerStream() const;
  void* getCoalesceBuffer();

 private:
  // create CUDA resources
  void createCudaResources();
  ncclComm_t comm_;
  int worldSize_, worldRank_;
  // CUDA stream in which NCCL calls run if in async mode
  std::shared_ptr<CUDAStream> reductionStream_;
  // CUDA stream in which cudaMemcpyAsync calls run if in contiguous mode
  std::shared_ptr<CUDAStream> workerStream_;
  // Buffer for storing copied gradients contiguously; exists on device memory
  void* coalesceBuffer_{nullptr};
  std::once_flag allocBuffer_;
};

bool isNonNegativeInteger(const std::string& s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

ncclDataType_t getNcclTypeForArray(const Tensor& arr) {
  switch (arr.type()) {
    case fl::dtype::f16:
      return ncclHalf;
    case fl::dtype::f32:
      return ncclFloat32;
    case fl::dtype::f64:
      return ncclFloat64;
    case fl::dtype::s32:
      return ncclInt32;
    case fl::dtype::s64:
      return ncclInt64;
      break;
    default:
      throw std::runtime_error("unsupported data type for allreduce with NCCL");
  }
}

} // namespace

void ncclCheck(ncclResult_t r);

void mpiCheck(int ec);

void allReduceCuda(
    const CUDAStream* bufferStream,
    void* ptr,
    const size_t count,
    const ncclDataType_t ncclType,
    const bool async,
    const bool contiguous);
} // namespace detail

void allReduce(Tensor& arr, bool async /* = false */) {
  if (!isDistributedInit()) {
    throw std::runtime_error("distributed environment not initialized");
  }
  ncclDataType_t type = detail::getNcclTypeForArray(arr);
  DevicePtr tensorPtr(arr);
  detail::allReduceCuda(
      &arr.stream().impl<CUDAStream>(),
      tensorPtr.get(),
      arr.elements(),
      type,
      async,
      /* contiguous = */ false);
}

void allReduceMultiple(
    std::vector<Tensor*> arrs,
    bool async /* = false */,
    bool contiguous /* = false */) {
  // Fast paths
  if (arrs.empty()) {
    return;
  }

  if (!contiguous) {
    // Use nccl groups to do everything in a single kernel launch
    NCCLCHECK(ncclGroupStart());
    for (auto& arr : arrs) {
      allReduce(*arr, async);
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
  size_t typeSize = fl::getTypeSize(arrs[0]->type());

  // Device ptrs from each array
  std::vector<std::pair<DevicePtr, size_t>> tensorPtrs;
  tensorPtrs.reserve(arrs.size());
  size_t totalEls{0};
  for (auto& arr : arrs) {
    totalEls += arr->elements();
    tensorPtrs.emplace_back(DevicePtr(*arr), arr->bytes());
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
  const auto& workerStream = ncclContext.getWorkerStream();

  const auto constTensors = std::vector<const Tensor*>(arrs.begin(), arrs.end());
  // Block the copy worker stream on Flashlight's active CUDA stream
  relativeSync(workerStream, constTensors);

  // In the worker stream, coalesce gradients into one large buffer so we
  // only need to call allReduce
  void* coalesceBuffer = ncclContext.getCoalesceBuffer();
  auto* cur = reinterpret_cast<char*>(coalesceBuffer);
  for (auto& entry : tensorPtrs) {
    FL_CUDA_CHECK(cudaMemcpyAsync(
        cur,
        entry.first.get(),
        entry.second,
        cudaMemcpyDeviceToDevice,
        workerStream.handle()));
    cur += entry.second;
  }

  // Now, call allReduce once on the entire copy buffer
  detail::allReduceCuda(
      &workerStream,
      coalesceBuffer,
      totalEls,
      ncclType,
      async,
      contiguous);

  // Block the worker stream's copy operations on allReduce operations that are
  // currently enqueued in the reduction stream
  if (async) {
    workerStream.relativeSync(ncclContext.getReductionStream());
  } else {
    relativeSync(workerStream, constTensors);
  }

  // Enqueue operations in the stream to copy back to each respective array from
  // the coalesce buffer
  cur = reinterpret_cast<char*>(coalesceBuffer);
  for (auto& entry : tensorPtrs) {
    FL_CUDA_CHECK(cudaMemcpyAsync(
        entry.first.get(),
        cur,
        entry.second,
        cudaMemcpyDeviceToDevice,
        workerStream.handle()));
    cur += entry.second;
  }
}

/**
 * Block future operations in all other CUDA streams on this device on
 * operations currently running in the NCCL [and worker] CUDA stream.
 */
void syncDistributed() {
  const auto& ncclContext = detail::NcclContext::getInstance();
  const auto& manager = DeviceManager::getInstance();
  const auto& activeCudaDevice = manager.getActiveDevice(DeviceType::CUDA);
  const auto& workerStream = ncclContext.getWorkerStream();
  const auto& reductionStream = ncclContext.getReductionStream();
  for (const auto& stream : activeCudaDevice.getStreams()) {
    if (stream.get() != &workerStream && stream.get() != &reductionStream) {
      stream->relativeSync(workerStream);
      stream->relativeSync(reductionStream);
    }
  }
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

void allReduceCuda(
    const CUDAStream* bufferStream,
    void* ptr,
    const size_t count,
    const ncclDataType_t ncclType,
    const bool async,
    const bool contiguous) {
  const CUDAStream* syncStream;
  auto& ncclContext = detail::NcclContext::getInstance();
  if (async) {
    syncStream = &ncclContext.getReductionStream();
  } else {
    syncStream = bufferStream;
  }

  // Synchronize with whatever CUDA stream is performing operations needed
  // pre-reduction. If we're in contiguous mode, we need the reduction stream to
  // wait for the copy in the worker stream to complete. If we're not in
  // CUDA stream.
  if (contiguous) {
    // block future reduction stream ops on the copy-worker stream
    syncStream->relativeSync(ncclContext.getWorkerStream());
  } else if (async) {
    syncStream->relativeSync(*bufferStream);
  }
  // don't synchronize streams if not async and not contiguous - the AF CUDA
  // stream does everything

  NCCLCHECK(ncclAllReduce(
      ptr,
      ptr,
      count,
      ncclType,
      ncclSum,
      ncclContext.getComm(),
      syncStream->handle()));
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

const CUDAStream& NcclContext::getReductionStream() const {
  return *reductionStream_;
}

const CUDAStream& NcclContext::getWorkerStream() const {
  return *workerStream_;
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
  // initialize
  // - dedicated NCCL CUDA stream to support async allReduce
  // - a third dedicated stream to asynchronously copy gradients
  //   into a coalesced form if using a contiguous allReduce

// Destroying the dedicated NCCL CUDA stream is a bit odd since the stream
// lives in a global NcclContext singleton. The CUDA driver shuts down when
// exit(0) is called, but the context may not be destroyed until
// afterwards, and destroying streams when the driver has already shut down is
// ungraceful. Manually destroying streams races against the driver, but in
// all cases, streams are destroyed when the driver shuts down, so don't
// destroy the stream by default.
#ifdef CUDA_STREAM_POOL_DESTROY_ON_SHUTDOWN
  reductionStream_ = CUDAStream::createManaged(detail::kDefaultStreamFlags);
  workerStream_ = CUDAStream::createManaged(detail::kDefaultStreamFlags);
#else
  reductionStream_ = CUDAStream::createUnmanaged(detail::kDefaultStreamFlags);
  workerStream_ = CUDAStream::createUnmanaged(detail::kDefaultStreamFlags);
#endif
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
  fl::setDevice(worldRank_ % std::stoi(maxDevicePerNode->second));

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

  fl::setDevice(worldRank_ % std::stoi(maxDevicePerNode->second));

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

  // Remove the temporary file created for initialization
  if (worldRank_ == 0) {
    fs.clear(kNcclKey);
  }

  createCudaResources();
}

NcclContext::~NcclContext() {
#ifdef NO_NCCL_COMM_DESTROY_HANDLE
// DEBUG : ncclCommDestroy disabled as it leads to segfault.
#else
  // finalizing NCCL
  NCCLCHECK(ncclCommDestroy(comm_));
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
