/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/StandAloneOptimizer.h"

#include <cstdio>
#include <fstream>
#include <vector>

#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/experimental/memory/AllocationLog.h"
#include "flashlight/fl/experimental/memory/allocator/ConfigurableMemoryAllocator.h"
#include "flashlight/fl/experimental/memory/optimizer/Simulator.h"

using namespace fl;

int main(int argc, char** argv) {
  Logging::setMaxLoggingLevel(INFO);
  VerboseLogging::setMaxLoggingLevel(0);

  if (argc < 2) {
    FL_LOG(fl::ERROR)
        << "Usage:" << argv[0]
        << " [fl_memory_manager_standalone_optimizer_config.json]";
    return writeTemplateConfigFile();
  }
  const std::string standaloneOptimizerConfigFilename = argv[1];
  FL_LOG(fl::INFO) << "Reading stand alone optimizer config from file="
                   << standaloneOptimizerConfigFilename;
  std::ifstream standaloneOptimizerConfigFile(
      standaloneOptimizerConfigFilename);
  if (!standaloneOptimizerConfigFile.is_open()) {
    FL_LOG(fl::ERROR) << "Failed to open stand alone optimizer config file="
                      << standaloneOptimizerConfigFilename;
    return -1;
  }
  StandAloneOptimizerConfig standaloneOptimizerConfig =
      StandAloneOptimizerConfig::loadJSon(standaloneOptimizerConfigFile);
  FL_LOG(fl::INFO) << "Stand alone optimizer config="
                   << standaloneOptimizerConfig.prettyString();

  std::vector<MemoryAllocatorConfiguration> initialAllocatorConfigs;

  for (int i = 2; i < argc; ++i) {
    const std::string initialConfigFilename = argv[i];
    // if ("fl_memory_manager_final_config_output.json" )
    if (standaloneOptimizerConfig.finalConfigFile == initialConfigFilename) {
      continue;
    }
    const std::string initialAllocatorConfigFilename =
        fullpath(standaloneOptimizerConfig.basePath, initialConfigFilename);
    FL_LOG(fl::INFO) << "Reading initial memory allocator config from file="
                     << initialAllocatorConfigFilename;
    std::ifstream initialAllocatorConfigFile(initialAllocatorConfigFilename);
    if (!initialAllocatorConfigFile.is_open()) {
      FL_LOG(fl::ERROR)
          << "Failed to open initial memory allocator config file="
          << initialAllocatorConfigFilename;
      return -1;
    }
    initialAllocatorConfigs.push_back(
        MemoryAllocatorConfiguration::loadJSon(initialAllocatorConfigFile));
    FL_LOG(fl::INFO) << "Initial memory allocator config="
                     << initialAllocatorConfigs.back().prettyString();
  }
  if (initialAllocatorConfigs.empty()) {
    FL_LOG(fl::ERROR)
        << "Please specify initial allocator configuration files.";
    return -1;
  }

  const std::string finalAllocatorConfigFilename = fullpath(
      standaloneOptimizerConfig.basePath,
      standaloneOptimizerConfig.finalConfigFile);
  FL_LOG(fl::INFO)
      << "Open for writing the final (optimized) memory allocator config file="
      << finalAllocatorConfigFilename;
  std::ofstream finalAllocatorConfigFile(finalAllocatorConfigFilename);
  if (!finalAllocatorConfigFile.is_open()) {
    FL_LOG(fl::ERROR)
        << "Failed to open final (optimized) memory allocator config file="
        << finalAllocatorConfigFilename;
    return -1;
  }

  const std::string allocationLogFilename = fullpath(
      standaloneOptimizerConfig.basePath,
      standaloneOptimizerConfig.allocationLog);
  FL_LOG(fl::INFO) << "Reading allocation log from file="
                   << allocationLogFilename;
  std::ifstream allocationLogFile(allocationLogFilename);
  if (!allocationLogFile.is_open()) {
    FL_LOG(fl::ERROR) << "Failed to open allocation log file="
                      << allocationLogFilename;
    return -1;
  }
  std::vector<AllocationEvent> allocationLog =
      LoadAllocationLog(allocationLogFile);
  FL_LOG(fl::INFO) << "Allocation log size=" << allocationLog.size();

  const std::string optimizerConfigFilename = fullpath(
      standaloneOptimizerConfig.basePath,
      standaloneOptimizerConfig.memoryOptimizerConfigurationFile);
  FL_LOG(fl::INFO) << "Reading memory allocator optimizer config from file="
                   << optimizerConfigFilename;
  std::ifstream memoryOptimizerConfigurationFile(optimizerConfigFilename);
  if (!memoryOptimizerConfigurationFile.is_open()) {
    FL_LOG(fl::ERROR)
        << "Failed to open memory allocator optimizer config file="
        << optimizerConfigFilename;
    return -1;
  }
  MemoryOptimizerConfiguration memoryOptimizerConfiguration =
      MemoryOptimizerConfiguration::loadJSon(memoryOptimizerConfigurationFile);
  FL_LOG(fl::INFO) << "Memory allocator optimizer config="
                   << memoryOptimizerConfiguration.prettyString();

  MemoryAllocatorConfiguration finalAllocatorConfig = randomNearSearchOptimizer(
      initialAllocatorConfigs, allocationLog, memoryOptimizerConfiguration);

  FL_LOG(fl::INFO) << "Final (optimized) config="
                   << finalAllocatorConfig.prettyString();
  FL_LOG(fl::INFO) << "Writing final memory allocator config to file="
                   << finalAllocatorConfigFilename;
  finalAllocatorConfig.saveJSon(finalAllocatorConfigFile);

  return 0;
}

namespace fl {
namespace {
#ifdef _WIN32
constexpr const char* kSeparator = "\\";
#else
constexpr const char* kSeparator = "/";
#endif

constexpr const char* kOptimizerConfigFilename =
    "fl_memory_manager_optimizer_config.json";

constexpr const char* kStandaloneOptimizerConfigFilename =
    "fl_memory_manager_standalone_optimizer_config.json";

} // namespace

std::string fullpath(const std::string& path, const std::string& filename) {
  if (filename.front() == kSeparator[0]) {
    return filename;
  }
  if (path.back() == kSeparator[0]) {
    return path + filename;
  }
  return path + kSeparator + filename;
}

int writeTemplateConfigFile() {
  std::stringstream optimizerConfigFilenameBuilder;
  optimizerConfigFilenameBuilder << std::tmpnam(nullptr) << '-'
                                 << kOptimizerConfigFilename;
  const std::string optimizerConfigFilename =
      optimizerConfigFilenameBuilder.str();
  FL_LOG(fl::INFO)
      << "Writing a MemoryOptimizerConfiguration templates for your convenience to="
      << optimizerConfigFilename;

  MemoryOptimizerConfiguration optimizerConfig;
  optimizerConfig.memorySize = 16521541376; // (15GB+396MB+173KB+768)
  optimizerConfig.oomEventCountWeight = 0.0;
  optimizerConfig.maxInternalFragmentationScoreWeight = 0.1;
  optimizerConfig.maxInternalFragmentationScoreThreshold = 0.0;
  optimizerConfig.maxExternalFragmentationScoreWeight = 0.2;
  optimizerConfig.maxExternalFragmentationScoreThreshold = 0.0;
  optimizerConfig.statsInBlocksMaxAllocatedRatioWeight = 0.3;
  optimizerConfig.statsInBlocksMaxAllocatedRatioThreashold = 0.0;
  optimizerConfig.numberOfIterations = 10;
  optimizerConfig.numberOfConfigsToGeneratePerIteration = 10;
  optimizerConfig.beamSize = 10;
  optimizerConfig.learningRate = 0;
  optimizerConfig.allocCountWeight = 0.5;
  optimizerConfig.normalizeWeights();

  std::ofstream optimizerConfigFile(optimizerConfigFilename);
  if (!optimizerConfigFile.is_open()) {
    FL_LOG(fl::ERROR)
        << "Failed to open for writing MemoryOptimizerConfiguration file="
        << optimizerConfigFilename;
    return -1;
  }
  optimizerConfig.saveJSon(optimizerConfigFile);

  std::stringstream standAloneOptimizerConfigFilenameBuilder;
  standAloneOptimizerConfigFilenameBuilder
      << std::tmpnam(nullptr) << '-' << kStandaloneOptimizerConfigFilename;
  const std::string standAloneOptimizerConfigFilename =
      standAloneOptimizerConfigFilenameBuilder.str();
  FL_LOG(fl::INFO)
      << "Writing a StandAloneOptimizerConfig template for your convenience to="
      << standAloneOptimizerConfigFilename;

  fl::StandAloneOptimizerConfig standaloneOptimizerConfig = {
      "/checkpoint/you/base/path/",
      "fl_memory_manager_initial_config_input.json",
      "fl_memory_manager_final_config_output.json",
      optimizerConfigFilename,
      "fl_memory_manager_allocation_log_output.csv",
      0.9};
  std::ofstream standaloneOptimizerConfigFileOstream(
      standAloneOptimizerConfigFilename);
  if (!standaloneOptimizerConfigFileOstream.is_open()) {
    FL_LOG(fl::ERROR) << "Failed to open for writing config file="
                      << standAloneOptimizerConfigFilename;
    return -1;
  }

  standaloneOptimizerConfig.saveJSon(standaloneOptimizerConfigFileOstream);
  return 0;
}

std::string StandAloneOptimizerConfig::prettyString() const {
  std::stringstream ss;
  ss << "StandAloneOptimizerConfig{basePath" << basePath
     << " initialConfigFile=" << initialConfigFile
     << " finalConfigFile=" << finalConfigFile
     << " allocationLog=" << allocationLog
     << " minAllocationRatioToDumpStats=" << minAllocationRatioToDumpStats
     << "}";
  return ss.str();
}

StandAloneOptimizerConfig StandAloneOptimizerConfig::loadJSon(
    std::istream& streamToConfig) {
  StandAloneOptimizerConfig config;
  try {
    cereal::JSONInputArchive archive(streamToConfig);
    archive(config);
  } catch (std::exception& ex) {
    FL_LOG(fl::ERROR)
        << "StandAloneOptimizerConfig::loadJSon() failed to load config with error="
        << ex.what();
    throw ex;
  }
  return config;
}

void StandAloneOptimizerConfig::saveJSon(std::ostream& saveConfigStream) const {
  try {
    cereal::JSONOutputArchive archive(saveConfigStream);
    archive(*this);
  } catch (std::exception& ex) {
    FL_LOG(fl::ERROR)
        << "StandAloneOptimizerConfig::saveJSon() failed to save config with error="
        << ex.what();
    throw ex;
  }
}

} // namespace fl
