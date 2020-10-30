/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/experimental/memory/AllocationLog.h"
#include "flashlight/fl/experimental/memory/allocator/ConfigurableMemoryAllocator.h"
#include "flashlight/fl/experimental/memory/optimizer/Optimizer.h"
#include "flashlight/fl/experimental/memory/optimizer/Simulator.h"

using namespace fl;

std::string resultFileName(std::string filename) {
  return filename + "-simulation-result.txt";
}

void simulate(
    size_t arenaSize,
    const std::string& allocationLogPath,
    std::vector<std::string> memoryAllocatorConfigPathVector) {
  {
    std::stringstream ss;
    ss << "simulate(arenaSize=" << prettyStringMemorySize(arenaSize)
       << " allocationLogPath=" << allocationLogPath
       << " memoryAllocatorConfigPathVector.size()="
       << memoryAllocatorConfigPathVector.size()
       << ") memoryAllocatorConfigPathVector:" << std::endl;
    for (const std::string& configFileName : memoryAllocatorConfigPathVector) {
      ss << configFileName << std::endl;
    }
    FL_LOG(fl::INFO) << ss.str();
  }

  std::ifstream allocationLogStream(allocationLogPath);
  if (!allocationLogStream.is_open()) {
    FL_LOG(fl::ERROR) << "simulate() failed to open allocation log file="
                      << allocationLogPath;
    return;
  }
  std::vector<AllocationEvent> allocationLog =
      LoadAllocationLog(allocationLogStream);
  if (allocationLog.empty()) {
    FL_LOG(fl::ERROR) << "simulate() empty allocation log file="
                      << allocationLogPath;
    return;
  }
  FL_LOG(fl::INFO) << "Allocation log size="
                   << prettyStringCount(allocationLog.size());

  std::vector<std::unique_ptr<MemoryAllocator>> allocators;
  std::vector<MemoryAllocator*> allocatorCopyVec;
  std::vector<MemoryAllocatorConfiguration> allocatorConfigVec;
  std::vector<std::ofstream> allocatorsFiles;
  for (const std::string& configFileName : memoryAllocatorConfigPathVector) {
    std::ifstream initConfigStream(configFileName);
    if (!initConfigStream.is_open()) {
      FL_LOG(fl::ERROR) << "Failed to read memory allocator config file="
                        << configFileName;
      return;
    }
    MemoryAllocatorConfiguration allocatorCnfig;
    try {
      allocatorCnfig = MemoryAllocatorConfiguration::loadJSon(initConfigStream);
    } catch (std::exception& ex) {
      FL_LOG(fl::ERROR) << "Failed to parse memory allocator config file="
                        << configFileName << " with error=" << ex.what();
      return;
    }
    void* arenaAddress = reinterpret_cast<void*>(0x10);
    std::unique_ptr<MemoryAllocator> allocator = CreateMemoryAllocator(
        allocatorCnfig, arenaAddress, arenaSize, /*logLevel=*/0);
    allocatorConfigVec.push_back(allocatorCnfig);
    allocatorCopyVec.push_back(allocator.get());
    allocators.push_back(std::move(allocator));

    allocatorsFiles.push_back(std::ofstream(resultFileName(configFileName)));
    FL_LOG(fl::INFO) << "file=" << resultFileName(configFileName)
                     << " is open=" << allocatorsFiles.back().is_open();
  }

  const unsigned int threadCount = std::min(
      static_cast<unsigned int>(memoryAllocatorConfigPathVector.size()),
      std::thread::hardware_concurrency());
  FL_LOG(fl::INFO) << "Creating a thread pool of size=" << threadCount;
  BlockingThreadPool threadPool(threadCount);
  std::vector<SimResult> simResults;
  try {
    FL_LOG(fl::INFO);
    simResults = simulateAllocatorsOnAllocationLog(
        threadPool, allocationLog, allocatorCopyVec);
    FL_LOG(fl::INFO) << " simResults.size()=" << simResults.size();
  } catch (std::exception& ex) {
    FL_LOG(fl::ERROR)
        << "simulateAllocatorsOnAllocationLog() failed with error="
        << ex.what();
  }

  std::vector<int> sortedIndices(simResults.size());
  std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
  auto comparator = [&simResults](int a, int b) {
    return simResults[a].timeElapsedNanoSec_ <
        simResults[b].timeElapsedNanoSec_;
  };
  std::sort(sortedIndices.begin(), sortedIndices.end(), comparator);

  std::stringstream summary;
  summary
      << "success,elapsed(sec),allocationsCount,maxExternalFragmentationScore,performanceCost,"
      << "maxInternalFragmentationScore,allocatorConfigFile,resultFileName"
      << std::endl;
  FL_LOG(fl::INFO) << "  memoryAllocatorConfigPathVector.size()="
                   << memoryAllocatorConfigPathVector.size();

  for (int i : sortedIndices) {
    const MemoryAllocator::Stats stats = allocatorCopyVec.at(i)->getStats();

    std::stringstream ss;
    summary << simResults.at(i).success_ << ", "
            << simResults.at(i).timeElapsedNanoSec_ << ", "
            << stats.allocationsCount << ", "
            << stats.maxExternalFragmentationScore << ", "
            << prettyStringCount(stats.performanceCost) << ", "
            << stats.maxInternalFragmentationScore << ", "
            << memoryAllocatorConfigPathVector.at(i) << ", "
            << resultFileName(memoryAllocatorConfigPathVector.at(i))
            << std::endl;

    try {
      std::ofstream& resultFile = allocatorsFiles.at(i);
      if (resultFile.is_open()) {
        resultFile << "simulate(arenaSize=" << prettyStringMemorySize(arenaSize)
                   << " allocationLogPath=" << allocationLogPath << ")"
                   << std::endl;
        resultFile << "allocatorConfigFile="
                   << memoryAllocatorConfigPathVector.at(i)
                   << " simResults=" << simResults.at(i).prettyString()
                   << " allocationsCount:" << stats.allocationsCount
                   << " maxExternalFragmentationScore="
                   << stats.maxExternalFragmentationScore << " performanceCost="
                   << prettyStringCount(stats.performanceCost)
                   << " maxInternalFragmentationScore="
                   << stats.maxInternalFragmentationScore << " resultFileName="
                   << resultFileName(memoryAllocatorConfigPathVector.at(i))
                   << std::endl;
        resultFile << "allocator=" << allocatorCopyVec.at(i)->prettyString()
                   << std::endl;
        resultFile << "allocatorCnfig="
                   << allocatorConfigVec.at(i).prettyString() << std::endl;
      } else {
        FL_LOG(fl::ERROR) << "Failed to write to resultFileName="
                          << resultFileName(
                                 memoryAllocatorConfigPathVector.at(i));
      }
    } catch (std::exception& ex) {
      FL_LOG(fl::ERROR) << ex.what();
    }
  }
  FL_LOG(fl::INFO) << "Summary of results:\n" << summary.str();
}

constexpr size_t kTypicalArenaSize16gb = 16523001856; // (15GB+397MB+576KB)

int main(int argc, char** argv) {
  Logging::setMaxLoggingLevel(INFO);
  VerboseLogging::setMaxLoggingLevel(0);

  if (argc < 4) {
    FL_LOG(fl::ERROR)
        << "Usage:" << argv[0]
        << " [arenasize] [path to allocation log csv file] "
        << "[path to memory config json file 1] ..  [path to memory config json file n]"
        << "\narenasize: use 16523001856 (15GB+397MB+576KB) for 16GB machines";
    return -1;
  }
  const size_t arenaSize = std::stol(argv[1]);
  if (arenaSize != kTypicalArenaSize16gb) {
    FL_LOG(fl::ERROR) << "Are you sure about the arena size? usueally it is:"
                      << kTypicalArenaSize16gb;
  }
  const std::string allocationLogPath = argv[2];
  std::vector<std::string> memoryAllocatorConfigPathVector;
  for (int i = 3; i < argc; ++i) {
    memoryAllocatorConfigPathVector.push_back(argv[i]);
  }

  simulate(arenaSize, allocationLogPath, memoryAllocatorConfigPathVector);

  return 0;
}
