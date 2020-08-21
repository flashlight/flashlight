/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <random>

#include <gtest/gtest.h>

#include "flashlight/common/Histogram.h"

using namespace fl;

namespace {

// Tests that FixedBucketSizeHistogram generate correct statistics for a
// normally distributed container of values. Checks that min,max,sum,mean,count
// and count per bucket makes sense.
TEST(FixedBucketSizeHistogram, NormalDistribution) {
  const int nValues = 10e6; // Random large number.
  const int nBuckes = 9; // Odd number of nuckets such that we have a bucket at
  // the center with max elements.
  const int mean = 100;
  const int stddev = 5;

  std::minstd_rand0 generator;
  std::normal_distribution<double> distribution(mean, stddev);

  std::vector<size_t> data(nValues);
  for (int i = 0; i < nValues; ++i) {
    data[i] = distribution(generator);
  }

  HistogramStats<size_t> hist =
      FixedBucketSizeHistogram<size_t>(data.begin(), data.end(), nBuckes);

  EXPECT_LT(hist.min, mean - stddev);
  EXPECT_GT(hist.max, mean + stddev);
  // Sum should be smaller than if all values where greater than the mean.
  EXPECT_LT(hist.sum, (nValues + 1) * mean);
  // Normal max should be greater than uniform distribution.
  EXPECT_GT(hist.maxNumValuesPerBucket, nValues / nBuckes);
  ASSERT_EQ(hist.buckets.size(), nBuckes);

  // Verify normal distribution.
  EXPECT_LT(hist.buckets[0].count, hist.buckets[1].count);
  EXPECT_LT(hist.buckets[1].count, hist.buckets[2].count);
  EXPECT_LT(hist.buckets[2].count, hist.buckets[3].count);
  EXPECT_LT(hist.buckets[3].count, hist.buckets[4].count);
  EXPECT_GT(hist.buckets[4].count, hist.buckets[5].count);
  EXPECT_GT(hist.buckets[5].count, hist.buckets[6].count);
  EXPECT_GT(hist.buckets[6].count, hist.buckets[7].count);
  EXPECT_GT(hist.buckets[7].count, hist.buckets[8].count);

  // Verify bounds span the range.
  EXPECT_EQ(hist.buckets[0].startInclusive, hist.min);
  for (int i = 0; i < (nBuckes - 1); ++i) {
    EXPECT_EQ(hist.buckets[i + 1].startInclusive, hist.buckets[i].endExclusive);
  }
  EXPECT_EQ(hist.buckets[nBuckes - 1].endExclusive, hist.max);

  std::cout << hist.prettyString() << std::endl;
}

// Tests that FixedBucketSizeHistogram generate correct statistics for a
// exponentially distributed container of values. Checks that
// min,max,sum,mean,count and count per bucket makes sense. Also tests
// correctness of an high-resolution histogram use case. High-resolution
// histogram makes sense for exponential distribution. It is an histogram of the
// bucket with the most elements in the first histogram.
TEST(FixedBucketSizeHistogram, ExponentialDistribution) {
  const int nValues = 10e6; // Random large number
  const int nBuckes = 12; // Random value
  const double multiplier =
      10e3; // Should be much bigger than 1 to map floats evenly on to integers.

  std::minstd_rand0 generator;
  std::exponential_distribution<double> distribution(0.1);

  std::vector<int> data(nValues);
  for (int i = 0; i < nValues; ++i) {
    data[i] = distribution(generator) * multiplier;
  }

  HistogramStats<int> hist =
      FixedBucketSizeHistogram<int>(data.begin(), data.end(), nBuckes);

  // Verify sanity of basic stats.
  ASSERT_EQ(hist.buckets.size(), nBuckes);
  EXPECT_EQ(hist.numValues, data.size());
  EXPECT_EQ(hist.min, 0);
  EXPECT_GT(hist.max, multiplier);
  // exponential max should be greater than uniform distribution.
  EXPECT_GT(hist.maxNumValuesPerBucket, nValues / nBuckes);

  // Verify exponential distribution.
  for (int i = 0; i < (nBuckes - 1); ++i) {
    EXPECT_GT(hist.buckets[i].count, hist.buckets[i + 1].count);
  }

  // Verify bounds span the range.
  EXPECT_EQ(hist.buckets[0].startInclusive, hist.min);
  for (int i = 0; i < (nBuckes - 1); ++i) {
    EXPECT_EQ(hist.buckets[i + 1].startInclusive, hist.buckets[i].endExclusive);
  }
  EXPECT_GE(hist.buckets[nBuckes - 1].endExclusive, hist.max);

  std::cout << hist.prettyString() << std::endl;

  // High-resolution histogram.
  const HistogramBucket<int>& largestCountBucket = hist.buckets[0];
  HistogramStats<int> hiResHist = FixedBucketSizeHistogram<int>(
      data.begin(),
      data.end(),
      nBuckes,
      largestCountBucket.startInclusive,
      largestCountBucket.endExclusive);

  // Verify sanity of basic stats.
  ASSERT_EQ(hiResHist.buckets.size(), nBuckes);
  EXPECT_GE(hiResHist.min, largestCountBucket.startInclusive);
  EXPECT_LE(hiResHist.max, largestCountBucket.endExclusive);
  // exponential max should be greater than uniform distribution.
  EXPECT_GT(hiResHist.maxNumValuesPerBucket, nValues / nBuckes);

  std::cout << hiResHist.prettyString() << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
