/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include <af/cuda.h>
#include <mpi.h>
#include <nccl.h>

#include "flashlight/common/CudaUtils.h"
#include "flashlight/distributed/DistributedApi.h"
#include "flashlight/distributed/backend/utils/FileStore.h"

#define NCCLCHECK(expr) ::fl::detail::ncclCheck((expr))
#define MPICHECK(expr) ::fl::detail::mpiCheck((expr))

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

void ncclCheck(ncclResult_t r);

void mpiCheck(int ec);

void allreduceCuda(void* ptr, size_t count, ncclDataType_t ncclType);
} // namespace detail

void allReduce(af::array& arr) {
  if (!isDistributedInit()) {
    throw std::runtime_error("distributed environment not initialized");
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
      throw std::runtime_error("unsupported data type for allreduce with NCCL");
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
    std::cout << "Initialized NCCL " << NCCL_MAJOR << "." << NCCL_MINOR
              << " successfully!\n";
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
