/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/experimental/memory/optimizer/Optimizer.h"

#include <algorithm>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "flashlight/fl/common/CppBackports.h"
#include "flashlight/fl/common/Histogram.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/experimental/memory/optimizer/Simulator.h"

namespace fl {
namespace {
// Permute the specified config dimention at a radius of at most ratio from
// the full spectrum of valid values.
// ratio: size of change -1..1
// dimention: 0-blockSize_, 1-maxAllocationSize_, 2-relativeSize_
bool permute(
    double ratio,
    size_t dimention,
    size_t subArenaConfigIndex,
    size_t arenaSize,
    MemoryAllocatorConfiguration* config) {
  std::stringstream ss;
  ss << "permute(ratio=" << ratio << " dimention=" << dimention
     << " subArenaConfigIndex=" << subArenaConfigIndex
     << " arenaSize=" << arenaSize << std::endl
     << "-->";
  if (subArenaConfigIndex >= config->subArenaConfiguration_.size()) {
    ss << " fail: subArenaConfigIndex=" << subArenaConfigIndex
       << " >= config->subArenaConfiguration_.size()="
       << config->subArenaConfiguration_.size();
    FL_LOG(fl::WARNING) << ss.str();
    return false;
  }
  if (dimention > 2) {
    ss << " fail dimention > 2";
    FL_LOG(fl::WARNING) << ss.str();
    return false;
  }
  const size_t minBlockSize = (1UL << config->alignmentNumberOfBits_);

  SubArenaConfiguration* subConfig =
      &config->subArenaConfiguration_[subArenaConfigIndex];
  const double subArenaSize =
      (subConfig->relativeSize_ * static_cast<double>(arenaSize));

  switch (dimention) {
    case 0: // blockSize_
    {
      const double maxNumberOfSteps = 10;
      int nStepsChange = ratio * maxNumberOfSteps;
      if (nStepsChange == 0) {
        nStepsChange = (ratio > 0) ? 1 : -1;
      }
      size_t& blockSize = subConfig->blockSize_;
      long newBlockSize = blockSize + nStepsChange * minBlockSize;
      if (newBlockSize < minBlockSize) {
        ss << "fails!! blockSize_=" << blockSize
           << " newBlockSize=" << newBlockSize << " < minSize ";
        FL_LOG(fl::INFO) << ss.str();
        return false;
      }
      if (newBlockSize > subArenaSize) {
        ss << "fails!! blockSize_=" << blockSize
           << " newBlockSize=" << newBlockSize << " > subArenaSize ";
        FL_LOG(fl::INFO) << ss.str();
        return false;
      }
      if (subArenaConfigIndex > 0) {
        if (newBlockSize <
            config->subArenaConfiguration_[subArenaConfigIndex - 1]
                .blockSize_) {
          ss << " fails!! subArenaConfigIndex=" << subArenaConfigIndex
             << " newBlockSize=" << newBlockSize << " < "
             << config->subArenaConfiguration_[subArenaConfigIndex - 1]
                    .blockSize_;
          FL_LOG(fl::WARNING) << ss.str();
          return false;
        }
      }

      {
        std::stringstream name;
        name << config->name_ << "-block-" << blockSize << '-' << newBlockSize;
        config->name_ = name.str();
      }

      size_t newMaxAllocationSize =
          (subConfig->maxAllocationSize_ / newBlockSize) * newBlockSize;
      ss << "permuting blockSize_  "
         << ((newBlockSize > blockSize) ? "up+" : "down-")
         << " old=" << blockSize << " new=" << newBlockSize
         << "  also adjusting maxAllocationSize_ old="
         << subConfig->maxAllocationSize_ << " new=" << newMaxAllocationSize;
      FL_LOG(fl::INFO) << ss.str();
      subConfig->maxAllocationSize_ = newMaxAllocationSize;
      blockSize = newBlockSize;
    } break;
    case 1: // maxAllocationSize_
    {
      const double maxNumberOfSteps = 100;
      int nStepsChange = ratio * maxNumberOfSteps;
      if (nStepsChange == 0) {
        nStepsChange = (ratio > 0) ? 1 : -1;
      }
      const size_t sizeChange = nStepsChange * subConfig->blockSize_;
      const size_t maxAllocationSize = subConfig->maxAllocationSize_;

      long newMaxAllocationSize = maxAllocationSize + sizeChange;
      if (newMaxAllocationSize < subConfig->blockSize_) {
        newMaxAllocationSize = maxAllocationSize - sizeChange;
      }
      if (newMaxAllocationSize < 0) {
        ss << "fails!! newMaxAllocationSize=" << newMaxAllocationSize
           << " < 0    subConfig->blockSize_=" << subConfig->blockSize_;
        FL_LOG(fl::WARNING) << ss.str();
        return false;
      }
      if (subArenaConfigIndex < config->subArenaConfiguration_.size() - 1) {
        if (newMaxAllocationSize >
            config->subArenaConfiguration_[subArenaConfigIndex + 1]
                .maxAllocationSize_) {
          ss << " fails!! subArenaConfigIndex=" << subArenaConfigIndex
             << " newMaxAllocationSize=" << newMaxAllocationSize << " > "
             << config->subArenaConfiguration_[subArenaConfigIndex + 1]
                    .maxAllocationSize_;
          FL_LOG(fl::WARNING) << ss.str();
          return false;
        }
      }
      // Ensure that max allocation newMaxAllocationSize is at least as large as
      // the one before.
      if (subArenaConfigIndex > 0) {
        if (newMaxAllocationSize <=
            config->subArenaConfiguration_[subArenaConfigIndex - 1]
                .maxAllocationSize_) {
          ss << " fail!! maxAllocationSize_=" << newMaxAllocationSize
             << " subArenaConfigIndex=" << subArenaConfigIndex
             << "  newMaxAllocationSize" << newMaxAllocationSize << " <= "
             << config->subArenaConfiguration_[subArenaConfigIndex - 1]
                    .maxAllocationSize_;
          FL_LOG(fl::INFO) << ss.str();
          return false;
        }
      }

      {
        std::stringstream name;
        name << config->name_ << "-alloc-" << maxAllocationSize << '-'
             << newMaxAllocationSize;
        config->name_ = name.str();
      }

      ss << "maxAllocationSize=" << maxAllocationSize
         << ((newMaxAllocationSize > maxAllocationSize) ? " up+ " : " down- ")
         << "  to new maxAllocationSize=" << newMaxAllocationSize;
      FL_LOG(fl::INFO) << ss.str();
      subConfig->maxAllocationSize_ = newMaxAllocationSize;
    } break;
    case 2: // relativeSize_
    {
      double& relativeSize = subConfig->relativeSize_;
      double newRelativeSize = relativeSize * (1.0 + ratio);
      if (relativeSize == 0) {
        if (ratio > 0) {
          newRelativeSize = ratio;
        }
      } else {
        int random = ((1.0 / ratio) * 7717.987);
        if (!(random % 11)) {
          newRelativeSize = 0;
        }
      }
      if (newRelativeSize == relativeSize) {
        ss << "fail!! newRelativeSize=" << newRelativeSize
           << "  == relativeSize";
        FL_LOG(fl::INFO) << ss.str();
        return false;
      }
      if (newRelativeSize < 0) {
        ss << "fail!! newRelativeSize=" << newRelativeSize << "  < 0";
        FL_LOG(fl::INFO) << ss.str();
        return false;
      }

      {
        std::stringstream name;
        name << config->name_ << "-relative-" << relativeSize << '-'
             << newRelativeSize;
        config->name_ = name.str();
      }

      try {
        config->normalize();
      } catch (std::exception& ex) {
        return false;
      }
      ss << "relativeSize=" << relativeSize
         << ((newRelativeSize > relativeSize) ? " up+ " : " down- ")
         << " to new relativeSize=" << newRelativeSize;
      FL_LOG(fl::INFO) << ss.str();
      relativeSize = newRelativeSize;
    } break;
    default:
      ss << "fail!! default";
      FL_LOG(fl::WARNING) << ss.str();
      return false;
  }
  // FL_LOG(fl::INFO) << ss.str();
  return true;
}

struct Result {
  Result() : loss_(std::numeric_limits<double>::max()), success_(false) {}

  std::string prettyString() const {
    std::stringstream ss;
    ss << "loss_=" << loss_ << " success_=" << success_ << " time_=" << time_
       << " stats_.allocationsCount=" << stats_.allocationsCount
       << " stats_.maxExternalFragmentationScore="
       << stats_.maxExternalFragmentationScore;
    return ss.str();
  }

  double loss_;
  MemoryAllocator::Stats stats_;
  bool success_;
  double time_;
};

} // namespace

void createConfigurationsNearInitial(
    const std::vector<MemoryAllocatorConfiguration>& initialConfigurations,
    const MemoryOptimizerConfiguration& optimizerConfig,
    double radius,
    std::map<MemoryAllocatorConfiguration, Result>* testedConfigurations,
    std::vector<MemoryAllocatorConfiguration>* newConfigs) {
  size_t newOnes = 0;
  size_t alreadyExists = 0;

  while (newOnes < optimizerConfig.numberOfConfigsToGeneratePerIteration) {
    FL_LOG(fl::INFO) << "createConfigurationsNearInitial() alreadyExists="
                     << alreadyExists << " newOnes=" << newOnes
                     << " numberOfConfigsToGeneratePerIteration="
                     << optimizerConfig.numberOfConfigsToGeneratePerIteration
                     << " initialConfigurations.size()="
                     << initialConfigurations.size();
    for (const MemoryAllocatorConfiguration& center : initialConfigurations) {
      size_t configsAroundCenter =
          optimizerConfig.numberOfConfigsToGeneratePerIteration /
          initialConfigurations.size();
      configsAroundCenter = std::max(configsAroundCenter, 1UL);
      std::vector<MemoryAllocatorConfiguration> nearConfigs =
          generateNearConfig(
              center, radius, configsAroundCenter, optimizerConfig.memorySize);
      for (const auto& newconfig : nearConfigs) {
        auto itr = testedConfigurations->find(newconfig);
        if (itr == testedConfigurations->end()) {
          (*testedConfigurations)[newconfig] = {};
          newConfigs->push_back(newconfig);
          ++newOnes;
        } else {
          ++alreadyExists;
        }
      }
      // FL_LOG(fl::INFO)
      // << "randomNearSearchOptimizer() single generateNearConfig()
      // alreadyExists="
      // << alreadyExists << " newOnes=" << newOnes;
    }
  }
}

MemoryAllocatorConfiguration randomNearSearchOptimizer(
    std::vector<MemoryAllocatorConfiguration> newCenters,
    const std::vector<AllocationEvent>& allocationLog,
    const MemoryOptimizerConfiguration& optimizerConfig) {
  FL_LOG(fl::INFO)
      << "randomNearSearchOptimizer() creating a thread pool of size="
      << std::thread::hardware_concurrency();
  BlockingThreadPool threadPool(std::thread::hardware_concurrency());

  std::map<MemoryAllocatorConfiguration, Result> configToResult;
  std::vector<MemoryAllocatorConfiguration> allConfigs;
  std::vector<MemoryAllocatorConfiguration> newConfigs = newCenters;
  std::vector<double> recentLossVector;
  std::vector<double> allLossVector;

  for (MemoryAllocatorConfiguration& conf : newConfigs) {
    configToResult[conf] = {};
  }

  double radius = 1;

  MemoryAllocatorConfiguration bestConfig;
  void* arenaAddress = reinterpret_cast<void*>(0x10);

  for (int i1 = 0; i1 < optimizerConfig.numberOfIterations; ++i1) {
    FL_LOG(fl::INFO) << "randomNearSearchOptimizer() iteration=" << i1;
    createConfigurationsNearInitial(
        newCenters, optimizerConfig, radius, &configToResult, &newConfigs);
    FL_LOG(fl::INFO) << "randomNearSearchOptimizer() iteration=" << i1
                     << " newConfigs.size()=" << newConfigs.size();

    std::vector<std::unique_ptr<MemoryAllocator>> allocatorObjects;
    std::vector<MemoryAllocator*> allocatorsSimulateAdapter;
    std::vector<std::string> errors;
    for (MemoryAllocatorConfiguration& config : newConfigs) {
      try {
        std::unique_ptr<MemoryAllocator> allocator = CreateMemoryAllocator(
            config,
            arenaAddress,
            optimizerConfig.memorySize,
            /*logLevel=*/0);
        allocatorsSimulateAdapter.push_back(allocator.get());
        allocatorObjects.push_back(std::move(allocator));
        allConfigs.push_back(config);
      } catch (std::exception& ex) {
        errors.push_back(ex.what());
      }
    }

    FL_LOG(fl::ERROR) << "Discarding invalid config with errors cnt="
                      << errors.size();

    std::vector<SimResult> simResults = simulateAllocatorsOnAllocationLog(
        threadPool, allocationLog, allocatorsSimulateAdapter);
    double totaltime = 0;
    size_t maxAllocCount = 0;
    for (int i2 = 0; i2 < allocatorsSimulateAdapter.size(); ++i2) {
      Result& result = configToResult[newConfigs.at(i2)];
      totaltime += simResults.at(i2).timeElapsedNanoSec_;
      result.stats_ = allocatorsSimulateAdapter[i2]->getStats();
      maxAllocCount = std::max(maxAllocCount, result.stats_.allocationsCount);
      result.success_ = simResults.at(i2).success_;
    }
    // double maxAllocCountFp = maxAllocCount;
    for (int i3 = 0; i3 < allocatorsSimulateAdapter.size(); ++i3) {
      Result& result = configToResult[newConfigs.at(i3)];
      result.time_ = (simResults.at(i3).timeElapsedNanoSec_ / totaltime);
      double numAllocRatio =
          (maxAllocCount + 1) - result.stats_.allocationsCount;

      result.loss_ = memoryAllocatorStatsLossFunction(
          result.success_,
          result.stats_,
          result.time_,
          numAllocRatio,
          optimizerConfig);
      recentLossVector.push_back(result.loss_ * 1000);
      allLossVector.push_back(result.loss_ * 1000);
    }

    std::vector<int> allConfigsIndexes(allConfigs.size());
    std::iota(allConfigsIndexes.begin(), allConfigsIndexes.end(), 0);

    std::sort(
        allConfigsIndexes.begin(),
        allConfigsIndexes.end(),
        [&configToResult, &allConfigs](int rhs, int lhs) {
          const Result& l = configToResult[allConfigs[lhs]];
          const Result& r = configToResult[allConfigs[rhs]];
          return l.loss_ > r.loss_;
        });

    std::vector<MemoryAllocatorConfiguration> allConfigsTmp;
    for (int i4 : allConfigsIndexes) {
      allConfigsTmp.push_back(allConfigs.at(i4));
    }
    allConfigs.swap(allConfigsTmp);

    FL_LOG(fl::INFO) << "randomNearSearchOptimizer() iteration=" << i1
                     << " completed. Summary:\n"
                     << "allConfigs size=" << allConfigs.size();
    HistogramStats<double> recentLossHist = FixedBucketSizeHistogram<double>(
        recentLossVector.begin(),
        recentLossVector.end(),
        kHistogramBucketCountPrettyString);
    recentLossVector.clear();
    HistogramStats<double> allLossVectorHist = FixedBucketSizeHistogram<double>(
        allLossVector.begin(),
        allLossVector.end(),
        kHistogramBucketCountPrettyString);

    FL_LOG(fl::INFO) << "recentLossHist\n:" << recentLossHist.prettyString();
    FL_LOG(fl::INFO) << "allLossVectorHist\n:"
                     << allLossVectorHist.prettyString();

    newConfigs.clear();
    radius *= (1.0 - optimizerConfig.learningRate);

    // Create new centers
    newCenters.clear();
    const size_t beamSize =
        std::min(allConfigs.size(), optimizerConfig.beamSize);
    FL_LOG(fl::INFO) << "Current beam state: beamSize=" << beamSize;
    for (int i5 = 0; i5 < beamSize; ++i5) {
      MemoryAllocatorConfiguration& config = allConfigs.at(i5);
      Result& result = configToResult[config];
      FL_LOG(fl::INFO) << "newCenters[i5=" << i5
                       << "] result=" << result.prettyString();
      newCenters.push_back(config);
    }
    FL_LOG(fl::INFO) << "Beam configs:";
    for (int i6 = 0; i6 < beamSize; ++i6) {
      MemoryAllocatorConfiguration& config = allConfigs.at(i6);
      FL_LOG(fl::INFO) << "newCenters[i6=" << i6 << "] "
                       << " config=" << config.prettyString();
      newCenters.push_back(config);
    }
  }

  return allConfigs.at(0);
}

void MemoryOptimizerConfiguration::normalizeWeights() {
  double sum = oomEventCountWeight + maxInternalFragmentationScoreWeight +
      maxExternalFragmentationScoreWeight +
      statsInBlocksMaxAllocatedRatioWeight;
  if (sum != 0) {
    oomEventCountWeight /= sum;
    maxInternalFragmentationScoreWeight /= sum;
    maxExternalFragmentationScoreWeight /= sum;
    statsInBlocksMaxAllocatedRatioWeight /= sum;
  }
}

std::string MemoryOptimizerConfiguration::prettyString() const {
  std::stringstream ss;
  ss << "MemoryOptimizerConfiguration{oomEventCountWeight="
     << oomEventCountWeight << " maxInternalFragmentationScoreWeight="
     << maxInternalFragmentationScoreWeight
     << " maxInternalFragmentationScoreThreshold="
     << maxInternalFragmentationScoreThreshold
     << " maxExternalFragmentationScoreWeight="
     << maxExternalFragmentationScoreWeight
     << " maxExternalFragmentationScoreThreshold="
     << maxExternalFragmentationScoreThreshold
     << " statsInBlocksMaxAllocatedRatioWeight="
     << statsInBlocksMaxAllocatedRatioWeight
     << " statsInBlocksMaxAllocatedRatioThreashold="
     << statsInBlocksMaxAllocatedRatioThreashold << "}";
  return ss.str();
}

MemoryOptimizerConfiguration MemoryOptimizerConfiguration::loadJSon(
    std::istream& streamToConfig) {
  MemoryOptimizerConfiguration config;
  cereal::JSONInputArchive archive(streamToConfig);
  archive(config);
  return config;
}

void MemoryOptimizerConfiguration::saveJSon(
    std::ostream& saveConfigStream) const {
  cereal::JSONOutputArchive archive(saveConfigStream);
  archive(*this);
}

double memoryAllocatorStatsLossFunction(
    bool success,
    const MemoryAllocator::Stats& stats,
    double timeElapsed,
    double numAllocRatio,
    const MemoryOptimizerConfiguration& config) {
  double val = stats.oomEventCount * config.oomEventCountWeight;
  val += stats.performanceCost * config.performanceCostWeight;
  if (stats.maxInternalFragmentationScore >=
      config.maxInternalFragmentationScoreThreshold) {
    val += stats.maxInternalFragmentationScore *
        config.maxInternalFragmentationScoreWeight;
  }
  if (stats.maxExternalFragmentationScore >=
      config.maxExternalFragmentationScoreThreshold) {
    val += stats.maxExternalFragmentationScore *
        config.maxExternalFragmentationScoreWeight * 10;
  }
  if (stats.statsInBlocks.maxAllocatedRatio() >=
      config.statsInBlocksMaxAllocatedRatioThreashold) {
    val += stats.statsInBlocks.maxAllocatedRatio() *
        config.statsInBlocksMaxAllocatedRatioWeight;
  }
  val += timeElapsed * 0.1;
  val += (numAllocRatio * 1e5);
  val += success ? 0 : 1e6;
  return val;
}

std::vector<MemoryAllocatorConfigurationLossOrdering>
orderMemoryAllocatorConfigByLoss(
    BlockingThreadPool& threadPool,
    const std::vector<MemoryAllocatorConfiguration>& haystack,
    const std::vector<AllocationEvent>& allocationLog,
    const MemoryOptimizerConfiguration& optimizerConfig) {
  FL_LOG(fl::INFO) << "orderMemoryAllocatorConfigByLoss(haystack=(size="
                   << haystack.size() << ") allocationLog=(size("
                   << allocationLog.size()
                   << ") optimizerConfig=" << optimizerConfig.prettyString();

  const int logLevel = 0;
  // Create an haystackOrder vector and point to corresponding allocator
  // configuration.
  std::vector<MemoryAllocatorConfigurationLossOrdering> haystackOrderVec(
      haystack.size());
  for (int i = 0; i < haystackOrderVec.size(); ++i) {
    haystackOrderVec[i].index = i;
  }

  void* arenaAddress = reinterpret_cast<void*>(0x10);
  // Simulate all configurations using a thread pool.
  for (int i2 = 0; i2 < haystack.size(); ++i2) {
    const MemoryAllocatorConfiguration& allocatorConfig = haystack[i2];
    MemoryAllocatorConfigurationLossOrdering* haystackOrder =
        &haystackOrderVec.at(i2);
    threadPool.enqueue([&allocationLog,
                        &allocatorConfig,
                        haystackOrder,
                        &optimizerConfig,
                        arenaAddress]() {
      std::unique_ptr<MemoryAllocator> allocator = CreateMemoryAllocator(
          allocatorConfig, arenaAddress, optimizerConfig.memorySize, logLevel);

      size_t arenaSize = allocator->getStats().statsInBlocks.arenaSize;

      if (arenaSize < 10) {
        FL_LOG(fl::ERROR) << "INVALID allocator!!!="
                          << allocator->prettyString();
        return;
      }

      bool success =
          simulateAllocatorOnAllocationLog(allocationLog, allocator.get());
      if (!success) {
        FL_LOG(fl::ERROR)
            << "Failed orderMemoryAllocatorConfigByLoss() haystackOrder->index="
            << haystackOrder->index;
        return;
      }
      haystackOrder->stats =
          fl::cpp::make_unique<MemoryAllocator::Stats>(allocator->getStats());
      haystackOrder->loss = memoryAllocatorStatsLossFunction(
          true, *haystackOrder->stats, 0, 0, optimizerConfig);
      haystackOrder->completedWithSuccess = true;
    });
  }
  threadPool.blockUntilAlldone();

  // Sort by loss value.
  std::sort(
      haystackOrderVec.begin(),
      haystackOrderVec.end(),
      [](MemoryAllocatorConfigurationLossOrdering& rhs,
         MemoryAllocatorConfigurationLossOrdering& lhs) {
        return lhs.loss > rhs.loss;
      });

  return haystackOrderVec;
}

std::vector<MemoryAllocatorConfiguration> generateNearConfig(
    const MemoryAllocatorConfiguration& center,
    double radius,
    size_t count,
    size_t arenaSize) {
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<int> subAllocatorIndexDistribution(
      0, center.subArenaConfiguration_.size() - 1);
  std::uniform_int_distribution<int> dimentionDistribution(0, 2);
  std::uniform_real_distribution<> ratioDistribution(-radius, radius);

  std::vector<MemoryAllocatorConfiguration> result;
  for (int i = 0; i < count; ++i) {
    result.push_back(center);
  };
  if (radius < 0.0001) {
    return result;
  }
  size_t mutationCount = 0;
  size_t failures = 0;
  FL_LOG(fl::INFO) << "generateNearConfig() mutationCount=" << mutationCount
                   << " count=" << count;
  while (mutationCount < count) {
    MemoryAllocatorConfiguration* config = &result.at(mutationCount);
    const size_t subArenaIndex = subAllocatorIndexDistribution(generator);
    const int dimention = dimentionDistribution(generator);
    const double ratio = ratioDistribution(generator);
    bool success = permute(ratio, dimention, subArenaIndex, arenaSize, config);
    if (success) {
      ++mutationCount;
      // FL_LOG(fl::INFO) << "config=" << config->prettyString();
    } else {
      ++failures;
    }
    FL_LOG(fl::WARNING) << "mutationCount=" << mutationCount
                        << " failures=" << failures;
  }
  return result;
}
}; // namespace fl
