/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fl {
namespace app {
namespace asr {
namespace sfx {

enum RandomPolicy { WITH_REPLACEMENT, WITHOUT_REPLACEMENT };
std::string randomPolicyToString(RandomPolicy policy);
RandomPolicy stringToRandomPolicy(const std::string& policy);

template <typename T>
class DatasetRandomiser {
 public:
  struct Config {
    RandomPolicy policy_ = WITH_REPLACEMENT;
    unsigned int randomSeed_ = std::mt19937::default_seed;
    std::string prettyString() const;
  };

  DatasetRandomiser(
      const DatasetRandomiser::Config& config,
      std::vector<T> dataset);

  size_t size() const;
  const T& getIndex(int index) const;
  const T& getRandom();
  std::string prettyString() const;

 private:
  Config conf_;
  std::mt19937 randomEngine_;
  std::uniform_int_distribution<> randomIndex_;
  std::vector<T> dataset_;
  std::vector<int> shuffle_;
  int count_;
};

template <typename T>
DatasetRandomiser<T>::DatasetRandomiser(
    const DatasetRandomiser::Config& config,
    std::vector<T> dataset)
    : conf_(config),
      randomEngine_(conf_.randomSeed_),
      randomIndex_(0, dataset.size() - 1),
      dataset_(std::move(dataset)),
      shuffle_(dataset_.size()),
      count_(0) {
  if (conf_.policy_ == WITHOUT_REPLACEMENT) {
    std::iota(shuffle_.begin(), shuffle_.end(), 0);
    const int n = shuffle_.size();
    // custom implementation of shuffle - https://stackoverflow.com/a/51931164
    for (int i = n; i >= 1; --i) {
      std::swap(shuffle_[i - 1], shuffle_[randomEngine_() % n]);
    }
  }
}

template <typename T>
const T& DatasetRandomiser<T>::getIndex(int getIndex) const {
  return dataset_[getIndex];
}

template <typename T>
const T& DatasetRandomiser<T>::getRandom() {
  if (conf_.policy_ == WITHOUT_REPLACEMENT) {
    return getIndex(shuffle_[count_++ % shuffle_.size()]);
  } else {
    return getIndex(randomIndex_(randomEngine_));
  }
}

template <typename T>
size_t DatasetRandomiser<T>::size() const {
  return dataset_.size();
}

template <typename T>
std::string DatasetRandomiser<T>::prettyString() const {
  std::stringstream ss;
  ss << "DatasetRandomiser{conf_=" << conf_.prettyString()
     << " dataset_.size()=" << dataset_.size() << " count_=" << count_ << "}";
  return ss.str();
}

template <typename T>
std::string DatasetRandomiser<T>::Config::prettyString() const {
  return "DatasetRandomiser::Config{policy_=" + randomPolicyToString(policy_) +
      " randomSeed_=" + std::to_string(randomSeed_) + "}";
}

} // namespace sfx
} // namespace asr
} // namespace app
} // namespace fl
