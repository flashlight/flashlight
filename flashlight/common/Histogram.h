/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fl {

// Write count into ss in a short hand form that fits in 5 chars. Always writes
// 5 chars into ss using as many leading spaces as required.
//
// Example:
// count=1234567  will be written as 1234K
// count=12345678 will be written as 12M
void shortFormatCount(std::stringstream& ss, size_t count);

// Write size into ss in a short hand form that fits in 5 chars. Always writes
// 5 chars into ss using as many leading spaces as required.
//
// Example:
// count=16384    (1 << 14) will be written as 16K
// count=1048576  (1 << 20) will be written as 1024K
// count=16777216 (1 << 24) will be written as 16M
void shortFormatMemory(std::stringstream& ss, size_t size);

using histValFmtFunc = std::function<void(std::stringstream&, size_t)>;

/**
 * Abstraction of generic histogram bucket. Used in the context of
 * HistogramStats
 */
template <typename T>
struct HistogramBucket {
  T startInclusive = 0; //! left boundary of the bucket.
  T endExclusive = 0; //! right boundary of the bucket.
  size_t count = 0; //! Number of elements in this bucket.88

  std::string prettyString(
      double countPerTick, // ratio of count/bar_length
      histValFmtFunc fromatCountIntoStream = shortFormatCount,
      histValFmtFunc fromatValuesIntoStream = shortFormatMemory) const;
};

/**
 * Generic data structure for representation of value set stats and histogram.
 */
template <typename T>
struct HistogramStats {
  // double bucketWidth = 0;
  T min = 0;
  T max = 0;
  T sum = 0;
  bool sumOverflow = false;
  double mean = 0;
  size_t numValues = 0;
  size_t maxNumValuesPerBucket = 0;
  std::vector<HistogramBucket<T>> buckets;

  std::string prettyString(
      size_t maxBarWidth = 50,
      histValFmtFunc fromatCountIntoStream = shortFormatCount,
      histValFmtFunc fromatValuesIntoStream = shortFormatMemory) const;
};

template <typename T>
bool isAdditionSafe(T a, T b) {
  if (a > (std::numeric_limits<T>::max() - b)) {
    return false;
  }
  if (std::is_signed<T>::value) {
    if (a < 0 && b < 0 && (a < (std::numeric_limits<T>::min() - b))) {
      return false;
    }
  }
  return true;
}

/**
 * Analyses a set of values from @begin to @end using histogram with fixed size
 * buckets.
 *
 * @param [begin, end] iterator of a value set of type T.
 * @param nBuckets number of buckets in the resulting histogram.
 * @param [clipMinValueInclusive,clipMaxValueExclusive] Consider only values
 * between the clipping bondenries
 */
template <typename T, typename Iterator>
HistogramStats<T> FixedBucketSizeHistogram(
    Iterator begin,
    Iterator end,
    size_t nBuckets,
    T clipMinValueInclusive = std::numeric_limits<T>::min(),
    T clipMaxValueExclusive = std::numeric_limits<T>::max()) {
  if (!nBuckets) {
    throw std::invalid_argument(
        "FixedBucketSizeHistogram(nBuckets=0) nBuckets "
        "must be a positive integer");
  }

  HistogramStats<T> stats;
  if (begin == end) {
    return stats;
  }

  stats.min = std::numeric_limits<T>::max();
  stats.max = std::numeric_limits<T>::min();
  stats.buckets.resize(nBuckets);

  // Calculate min/max, sum, ands mean
  double simpleMovingAverage = 0.0;
  for (auto itr = begin; itr != end; ++itr) {
    if ((*itr < clipMinValueInclusive) || (*itr >= clipMaxValueExclusive)) {
      continue;
    }
    if (!stats.sumOverflow) {
      if (isAdditionSafe(stats.sum, *itr)) {
        stats.sum += *itr;
      } else {
        stats.sumOverflow = true;
      }
    }

    stats.min = std::min(stats.min, *itr);
    stats.max = std::max(stats.max, *itr);
    double denominator = static_cast<double>(stats.numValues + 1);
    double ratio = stats.numValues / denominator;
    simpleMovingAverage = simpleMovingAverage * ratio + (*itr / denominator);
    ++stats.numValues;
  }
  stats.mean = simpleMovingAverage;

  // Calculate bucket size
  long range = stats.max - stats.min;
  double bucketWidth = range / nBuckets;
  if (range == 0 || bucketWidth == 0) {
    stats.buckets[0].count = stats.numValues;
    stats.maxNumValuesPerBucket = stats.numValues;
    return stats;
  }

  // Calculate count per bucket
  stats.maxNumValuesPerBucket = 0;
  for (auto itr = begin; itr != end; ++itr) {
    if (*itr < clipMinValueInclusive || *itr > clipMaxValueExclusive) {
      continue;
    }
    double index =
        std::round(static_cast<double>(*itr - stats.min) / bucketWidth);
    size_t intIndex = std::min(static_cast<size_t>(index), nBuckets - 1);

    HistogramBucket<T>& bucket = stats.buckets[intIndex];
    ++bucket.count;

    stats.maxNumValuesPerBucket =
        std::max(stats.maxNumValuesPerBucket, bucket.count);
  }

  // Set bucket start/end
  int i = 0;
  for (auto& bucket : stats.buckets) {
    bucket.startInclusive = stats.min + bucketWidth * i;
    bucket.endExclusive = stats.min + bucketWidth * (i + 1);
    ++i;
  }
  // Fix possible finite precision algebra mistakes
  stats.buckets.rbegin()->endExclusive = stats.max;

  return stats;
}

template <typename T>
std::string HistogramBucket<T>::prettyString(
    double countPerTick,
    histValFmtFunc fromatCountIntoStream,
    histValFmtFunc fromatValuesIntoStream) const {
  std::stringstream ss;
  ss << '[';
  fromatValuesIntoStream(ss, startInclusive);
  ss << '-';
  fromatValuesIntoStream(ss, endExclusive);
  ss << "] ";
  fromatCountIntoStream(ss, count);
  ss << ": ";
  const double numTicks = static_cast<double>(count) / countPerTick;
  for (int i = 0; i < std::round(numTicks); ++i) {
    ss << "*";
  }
  return ss.str();
};

template <typename T>
std::string HistogramStats<T>::prettyString(
    size_t maxBarWidth,
    histValFmtFunc fromatCountIntoStream,
    histValFmtFunc fromatValuesIntoStream) const {
  std::stringstream ss;
  ss << "HistogramStats{"
     << " min=[";
  fromatValuesIntoStream(ss, min);
  ss << "] max_=[";
  fromatValuesIntoStream(ss, max);
  ss << "] sum=[";
  if (sumOverflow) {
    ss << "overflow";
  } else {
    fromatCountIntoStream(ss, sum);
  }
  ss << "] mean=[";
  fromatValuesIntoStream(ss, mean);
  ss << "] numValues=[";
  fromatCountIntoStream(ss, numValues);
  ss << "] numBuckets=[" << buckets.size() << "]\n";
  if (buckets.size() > 1) {
    double countPerTick =
        static_cast<double>(maxNumValuesPerBucket) / maxBarWidth;
    for (const auto& bucket : buckets) {
      ss << bucket.prettyString(
          countPerTick, fromatCountIntoStream, fromatValuesIntoStream);
      ss << std::endl;
    }
  }
  return ss.str();
}

} // namespace fl
