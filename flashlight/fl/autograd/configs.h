
#pragma once

#include <memory>
#include <unordered_map>

#include <arrayfire.h>

namespace fl {
namespace detail {

/**
 * A generic data structure to save the configurations for a given algorithm.
 */
struct AlgoConfigs {
  uint algo;
  size_t memory;
  uint mathType;
  AlgoConfigs(uint algo, size_t memory, uint mathType)
      : algo(algo), memory(memory), mathType(mathType) {}
  AlgoConfigs() {}
};

/**
 * Map of data size, batch size, and type to algorithm configuration.
 */
class AlgoConfigMap {
 public:
  bool find(uint inputX, uint batchSize, uint type) {
    if (configs_.find(inputX) == configs_.end()) {
      return false;
    }
    if (configs_.at(inputX).find(batchSize) == configs_.at(inputX).end()) {
      return false;
    }
    if (configs_.at(inputX).at(batchSize).find(type) ==
        configs_.at(inputX).at(batchSize).end()) {
      return false;
    }
    return true;
  }

  AlgoConfigs get(uint inputX, uint batchSize, uint type) {
    return configs_.at(inputX).at(batchSize).at(type);
  }

  void set(uint inputX, uint batchSize, uint type, AlgoConfigs configs) {
    configs_[inputX][batchSize][type] = configs;
  }

 private:
  std::unordered_map<
      uint,
      std::unordered_map<uint, std::unordered_map<uint, AlgoConfigs>>>
      configs_;
};

/**
 * `ConvAlgoConfigs` is designed to incorporate all the configurations that are
 * required for a convolutional layers.
 * Hashmaps provide a mapping between layer shapes and the most performant
 * algorithm for those shapes. The implicit assumption is that input sizes may
 * differ in two dimensions: length and batch size. The former can be observed
 * in ASR application and the latter can occur while switching from training to
 * validation.
 */
struct ConvAlgoConfigs {
  AlgoConfigMap fwd;
  AlgoConfigMap bwdFilter;
  AlgoConfigMap bwdData;
};

using ConvAlgoConfigsPtr = std::shared_ptr<ConvAlgoConfigs>;

} // namespace detail
} // namespace fl
