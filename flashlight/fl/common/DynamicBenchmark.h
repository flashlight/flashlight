/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arrayfire.h>

#include "flashlight/fl/common/CppBackports.h"

namespace fl {

/**
 * A placeholder class that facilitates using DynamicBenchmark without generics;
 * that class holds onto a pointer of this type and is specialized by the user
 * in implementation-specific code.
 *
 * This type shouldn't be directly constructed.
 */
struct DynamicBenchmarkOptionsBase {
  virtual ~DynamicBenchmarkOptionsBase() = default;

  virtual void accumulateTimeToCurrentOption(double, bool = true) {
    throw std::logic_error(
        "DynamicBenchmarkOptionsBase::accumulateTimeToCurrentOption "
        "- unimplemented");
  }

  virtual bool timingsComplete() {
    throw std::logic_error(
        "DynamicBenchmarkOptionsBase::timingsComplete "
        "- unimplemented");
  }

 protected:
  // Not intended for construction
  DynamicBenchmarkOptionsBase() = default;
};

/**
 * Tracks performance of several different options in a dynamic benchmark. Keeps
 * track of:
 * - How many times an option has been used (ensures that all options are
 *   benchmarked the same number of times)
 * - The cumulative timings associated with a particular option.
 * - The "current" option, which is the option to which timings will be
 *   accumulated, and provides the option with the lowest timing/best
 *   performance when timings are complete.
 */
template <typename T>
struct DynamicBenchmarkOptions : DynamicBenchmarkOptionsBase {
  /**
   * Constructs an instance given a vector of options of specified type. The
   * options are assumed to be distinct since benchmarks options are
   * determined by index.
   *
   * @param[in] options vector of options to use
   * @param[in] benchCount the number of times to benchmark each option before
   * fixing on the optimal option
   */
  DynamicBenchmarkOptions(std::vector<T> options, size_t benchCount)
      : options_(options), benchCount_(benchCount) {
    if (options_.empty()) {
      throw std::invalid_argument(
          "DynamicBenchmarkOptions: "
          "Options must be passed vector with at least one element");
    }
    reset();
  }

  /**
   * Constructs an instance given a set of options.
   *
   * @param[in] options a set of options to use
   * @param[in] benchCount the number of times to benchmark each option before
   * fixing on the optimal option
   */
  DynamicBenchmarkOptions(
      fl::cpp::fl_unordered_set<T> options,
      size_t benchCount)
      : DynamicBenchmarkOptions(
            std::vector<T>(options.begin(), options.end()),
            benchCount) {}

  /**
   * Gets the current option; updates the current state.
   *
   * If each option hasn't been used/timed as many times as the max count, pick
   * the first option that hasn't been timed to the maximum count. If all
   * timings are complete, choose the optimal timing.
   *
   * @return the current option.
   */
  T updateState() {
    if (!timingsComplete_) {
      for (size_t i = 0; i < options_.size(); ++i) {
        if (counts_[i] < benchCount_) {
          currentOptionIdx_ = i;
          return options_[i];
        }
      }
      timingsComplete_ = true;

      // All options have been benchmarked with the max count - pick the one
      // with the lowest time
      size_t minTimeOptionIdx{0};
      for (size_t i = 0; i < options_.size(); ++i) {
        if (times_[i] < times_[minTimeOptionIdx]) {
          minTimeOptionIdx = i;
        }
      }
      currentOptionIdx_ = minTimeOptionIdx;
    }
    return options_[currentOptionIdx_];
  }

  /**
   * Gets the options' current value. This is deterministically computed and
   * only changes as per calls to `accumulateTimeToCurrentOption` that may
   * increment the count
   *
   * @return T the current option.
   */
  T currentOption() {
    return updateState();
  }

  /**
   * @return whether or not this options' timings are complete.
   */
  bool timingsComplete() override {
    updateState();
    return timingsComplete_;
  }

  /**
   * Adds time to the current option tally.
   *
   * @param[in] time duration to add
   * @param[in] incrementCount whether or not to increment the benchmark talley
   * for the option. This facilitates timing options by using results from
   * discontinuous functions
   */
  void accumulateTimeToCurrentOption(double time, bool incrementCount = true)
      override {
    if (timingsComplete()) {
      throw std::invalid_argument(
          "Options::accumulateTimeToCurrentOption: "
          "Tried to accumulate time when benchmarking is complete");
    }
    updateState();
    times_[currentOptionIdx_] += time;
    if (incrementCount) {
      counts_[currentOptionIdx_]++;
    }
  }

  /**
   * Resets options state to the default. Clears timings and counts.
   */
  void reset() {
    for (size_t i = 0; i < options_.size(); ++i) {
      counts_[i] = 0;
      times_[i] = 0.;
    }
    timingsComplete_ = false;
    currentOptionIdx_ = 0;
  }

 private:
  const std::vector<T> options_;
  const size_t benchCount_{0};

  bool timingsComplete_{false};
  int currentOptionIdx_{0}; // first option is the default
  // Number of times the option at each index has been timed
  std::unordered_map<size_t, int> counts_;
  // Accumulated times for each option
  std::unordered_map<size_t, double> times_;
};

/**
 * Dynamically times the execution of closures given some options given as
 * a specialization of a derived `DynamicBenchmarkOptionsBase`. This facilitates
 * benchmarking various configurations at runtime rather than pre-determining
 * configurations based on detected hardware.
 */
class DynamicBenchmark {
 public:
  explicit DynamicBenchmark(
      std::shared_ptr<DynamicBenchmarkOptionsBase> options)
      : options_(options) {}

  virtual ~DynamicBenchmark() = default;

  /**
   * Audits a dynamic benchmark function. Acccumulates times based on this
   * DynamicBenchmark's options' currently-active option.
   *
   * If the timings are complete for the benchmark options, simply executes the
   passed function. Calls `af::sync()` before and after function execution to
   get an accurate count.
   *
   * @param[in] function the function to benchmark
   */
  void audit(const std::function<void()>& function, bool incrementCount = true);

  /**
   * Gets the benchmarks' underlying `DynamicBenchmarkOptionsBase` instance.
   *
   * @return a pointer to the underlying options.
   */
  template <typename T>
  std::shared_ptr<T> getOptions() const {
    return std::static_pointer_cast<T>(options_);
  }

  /**
   * Sets global benchmark mode. If benchmark mode is on, all
   * `DynamicBenchmark`s will run normally. If benchmark mode is off, calling
   * `DynamicBenchmark::audit` with a given closure will simply execute the
   * closure.
   *
   * @param[in] mode the new value of benchmark mode
   */
  static void setBenchmarkMode(bool mode);

  /**
   * @return whether benchmark mode is globally enabled
   */
  static bool getBenchmarkMode();

 private:
  // Starts the benchmark timer
  void start();
  // Stops the benchmark timer, accumulates times to the current option
  void stop(bool incrementCount);

  std::shared_ptr<DynamicBenchmarkOptionsBase> options_;
  // Timer for current benchmark iteration
  af::timer currentTimer_;

  // Global fl benchmark mode - if off, no benchmarks will run, and audited
  // functions will be run directly without timings
  static bool benchmarkMode_;
};

// Specific benchmark implementations
namespace detail {

struct ConvBenchmarks {
  std::shared_ptr<DynamicBenchmark> bwdFilterBenchmark;
  std::shared_ptr<DynamicBenchmark> bwdDataBenchmark;
  std::shared_ptr<DynamicBenchmark> bwdBiasBenchmark;
};

} // namespace detail

} // namespace fl
