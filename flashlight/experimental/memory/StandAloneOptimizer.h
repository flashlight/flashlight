/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * The configurable memory allocator is obtained by calling:
 * std::unique_ptr<MemoryAllocator> CreateMemoryAllocator(config)
 * Config defines a a set of allocators assembled in a CompositeMemoryAllocator.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include "flashlight/flashlight/experimental/memory/AllocationLog.h"
#include "flashlight/flashlight/experimental/memory/allocator/ConfigurableMemoryAllocator.h"
#include "flashlight/flashlight/experimental/memory/allocator/MemoryAllocator.h"
#include "flashlight/flashlight/experimental/memory/optimizer/Optimizer.h"

namespace fl {
std::string fullpath(const std::string& path, const std::string& filename);

// Return success status similar to main()
// Write template config files to help the user to quickly customize one.
// Files are saved to safe paths created using std::tmpnam()
int writeTemplateConfigFile();

struct StandAloneOptimizerConfig {
  std::string basePath;
  std::string initialConfigFile;
  std::string finalConfigFile;
  std::string memoryOptimizerConfigurationFile;
  std::string allocationLog;
  double minAllocationRatioToDumpStats;

  std::string prettyString() const;

  static StandAloneOptimizerConfig loadJSon(std::istream& streamToConfig);
  void saveJSon(std::ostream& saveConfigStream) const;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::make_nvp("basePath", basePath),
       cereal::make_nvp("initialConfigFile", initialConfigFile),
       cereal::make_nvp("finalConfigFile", finalConfigFile),
       cereal::make_nvp(
           "memoryOptimizerConfigurationFile",
           memoryOptimizerConfigurationFile),
       cereal::make_nvp("allocationLog", allocationLog),
       cereal::make_nvp(
           "minAllocationRatioToDumpStats", minAllocationRatioToDumpStats));
  }
};

std::unique_ptr<MemoryAllocator> simulate(
    const StandAloneOptimizerConfig& config);

} // namespace fl
