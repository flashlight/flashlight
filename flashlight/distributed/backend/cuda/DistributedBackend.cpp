/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/distributed/DistributedApi.h>

#include <algorithm>
#include <cstring>

#include <af/cuda.h>
#include <glog/logging.h>
#include <mpi.h>
#include <nccl.h>

#include <flashlight/common/CudaUtils.h>
#include <flashlight/distributed/backend/utils/FileStore.h>

#define NCCLCHECK(cmd)                                               \
  do {                                                               \
    ncclResult_t r = cmd;                                            \
    if (r != ncclSuccess) {                                          \
      LOG(FATAL) << "Failed, NCCL error: " << ncclGetErrorString(r); \
    }                                                                \
  } while (0)

#define MPICHECK(cmd)                           \
  do {                                          \
    int e = cmd;                                \
    if (e != MPI_SUCCESS) {                     \
      LOG(FATAL) << "Failed, MPI error: " << e; \
    }                                           \
  } while (0)

namespace fl {

namespace detail {

namespace {

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

 private:
  ncclComm_t comm_;
  int worldSize_, worldRank_;
};

bool isNonNegativeInteger(const std::string& s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}
} // namespace

void allreduceCuda(void* ptr, size_t count, ncclDataType_t ncclType);
} // namespace detail

void allReduce(af::array& arr) {
  if (!isDistributedInit()) {
    LOG(FATAL) << "Distributed environment not initialized";
  }
  void* arrPtr = arr.device<void>();
  switch (arr.type()) {
    case af::dtype::f32:
      detail::allreduceCuda(arrPtr, arr.elements(), ncclFloat32);
      break;
    case af::dtype::f64:
      detail::allreduceCuda(arrPtr, arr.elements(), ncclFloat64);
      break;
    case af::dtype::s32:
      detail::allreduceCuda(arrPtr, arr.elements(), ncclInt32);
      break;
    case af::dtype::s64:
      detail::allreduceCuda(arrPtr, arr.elements(), ncclInt64);
      break;
    default:
      LOG(FATAL) << "Unsupported Arrayfire type for allreduce with Nccl";
  }
  arr.unlock();
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
    LOG(FATAL) << "Distributed init is being called more than once";
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
    LOG(FATAL) << "Unknown init method for distributed";
  }
  detail::DistributedInfo::getInstance().isInitialized_ = true;
  detail::DistributedInfo::getInstance().backend_ = DistributedBackend::NCCL;
}

namespace detail {

void allreduceCuda(void* ptr, size_t count, ncclDataType_t ncclType) {
  // TODO: Run NCCL and AF in different streams
  auto s = cuda::getActiveStream(); // assumes current device id
  NCCLCHECK(ncclAllReduce(
      ptr,
      ptr,
      count,
      ncclType,
      ncclSum,
      detail::NcclContext::getInstance().getComm(),
      s));
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

/* static */ NcclContext& NcclContext::getInstance() {
  static NcclContext ncclCtx;
  return ncclCtx;
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
    LOG(FATAL) << "Invalid params for distributed environment initialization";
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

  if (worldRank_ == 0) {
    LOG(INFO) << "Initialized NCCL successfully! Compiled with NCCL "
              << NCCL_MAJOR << "." << NCCL_MINOR;
  }
}

void NcclContext::initWithFileSystem(
    int worldRank,
    int worldSize,
    const std::unordered_map<std::string, std::string>& params) {
  auto filePath = params.find(DistributedConstants::kFilePath);
  auto maxDevicePerNode = params.find(DistributedConstants::kMaxDevicePerNode);

  if (filePath == params.end() || maxDevicePerNode == params.end()) {
    LOG(FATAL) << "Invalid params for distributed environment initialization";
  }

  worldRank_ = worldRank;
  worldSize_ = worldSize;

  ncclUniqueId id;

  af::setDevice(worldRank_ % std::stoi(maxDevicePerNode->second));

  // get NCCL unique ID at rank 0 and broadcast it to all others
  if (worldRank_ == 0) {
    ncclGetUniqueId(&id);
  }
  LOG_IF(FATAL, filePath->second.empty())
      << "Invalid args. Failed to initialize NCCL context.";

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

  if (worldRank_ == 0) {
    LOG(INFO) << "Initialized NCCL successfully! Compiled with NCCL "
              << NCCL_MAJOR << "." << NCCL_MINOR;
  }
}

NcclContext::~NcclContext() {
#ifdef NO_NCCL_COMM_DESTROY_HANDLE
  // DEBUG : ncclCommDestroy disabled as it leads to segfault.
#else
  // finalizing NCCL
  NCCLCHECK(ncclCommDestroy(comm_));
#endif
  if (DistributedInfo::getInstance().initMethod_ == DistributedInit::MPI) {
    // finalizing MPI
    MPICHECK(MPI_Finalize());
  }
}
} // namespace
} // namespace detail

} // namespace fl
