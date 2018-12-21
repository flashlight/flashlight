/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <thread>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include <flashlight/dataset/datasets.h>

using namespace fl;

bool allClose(
    const af::array& a,
    const af::array& b,
    const double precision = 1e-5) {
  if ((a.numdims() != b.numdims()) || (a.dims() != b.dims())) {
    return false;
  }
  return (af::max<double>(af::abs(a - b)) < precision);
}

TEST(DatasetTest, TensorDataset) {
  std::vector<af::array> tensormap = {af::randu(100, 200, 300),
                                      af::randu(150, 300)};
  TensorDataset tensords(tensormap);

  // Check `size` method
  ASSERT_EQ(tensords.size(), 300);

  // Values using `get` method
  auto ff1 = tensords.get(10);
  ASSERT_EQ(ff1.size(), 2);
  ASSERT_TRUE(allClose(ff1[0], tensormap[0](af::span, af::span, 10)));
  ASSERT_TRUE(allClose(ff1[1], tensormap[1](af::span, 10)));
}

TEST(DatasetTest, TranformDataset) {
  // first create a tensor dataset
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  auto scaleAndAdd = [](const af::array& a) { return af::sin(a) + 1.0; };
  TransformDataset transformds(tensords, {scaleAndAdd});

  // Check `size` method
  ASSERT_TRUE(transformds.size() == 300);

  // Values using `get` method
  auto ff1 = transformds.get(10);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], af::sin(tensormap[0](af::span, af::span, 10)) + 1.0));
}

TEST(DatasetTest, BatchDataset) {
  // first create a tensor dataset
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  BatchDataset batchds(tensords, 7, BatchDatasetPolicy::INCLUDE_LAST);

  // Check `size` method
  ASSERT_EQ(batchds.size(), 43);

  // Values using `get` method
  auto ff1 = batchds.get(42);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], tensormap[0](af::span, af::span, af::seq(294, 299))));

  ff1 = batchds.get(10);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], tensormap[0](af::span, af::span, af::seq(70, 76))));
}

TEST(DatasetTest, ShuffleDataset) {
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);
  ShuffleDataset shuffleds(tensords);

  // Check `size` method
  ASSERT_EQ(shuffleds.size(), 300);

  // Values using `get` method
  auto ff1 = shuffleds.get(10);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_FALSE(allClose(ff1[0], tensormap[0](af::span, af::span, 10)));
}

TEST(DatasetTest, ResampleDataset) {
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);
  auto permfn = [](int64_t n) { return (n + 5) % 300; };
  ResampleDataset resampleleds(tensords, permfn);

  // Check `size` method
  ASSERT_EQ(resampleleds.size(), 300);

  auto ff1 = resampleleds.get(10);
  ASSERT_TRUE(allClose(ff1[0], tensormap[0](af::span, af::span, 15)));
  ASSERT_FALSE(allClose(ff1[0], tensormap[0](af::span, af::span, 10)));
}

TEST(DatasetTest, ConcatDataset) {
  auto tensor1 = af::randu(100, 200, 100);
  auto tensor2 = af::randu(100, 200, 200);
  std::vector<af::array> tensormap1 = {tensor1};
  auto tensords1 = std::make_shared<TensorDataset>(tensormap1);
  std::vector<af::array> tensormap2 = {tensor2};
  auto tensords2 = std::make_shared<TensorDataset>(tensormap2);
  ConcatDataset concatds({tensords1, tensords2});

  // Check `size` method
  ASSERT_TRUE(concatds.size() == 300);

  auto ff1 = concatds.get(100);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(allClose(ff1[0], tensor2(af::span, af::span, 0)));
  ff1 = concatds.get(299);
  ASSERT_TRUE(allClose(ff1[0], tensor2(af::span, af::span, 199)));
  ff1 = concatds.get(0);
  ASSERT_TRUE(allClose(ff1[0], tensor1(af::span, af::span, 0)));
  ff1 = concatds.get(10);
  ASSERT_TRUE(allClose(ff1[0], tensor1(af::span, af::span, 10)));
}

TEST(DatasetTest, DatasetIterator) {
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  auto scaleAndAdd = [](const af::array& a) { return af::sin(a) + 1.0; };
  TransformDataset transformds(tensords, {scaleAndAdd});

  int idx = 0;
  for (auto& sample : transformds) {
    (void)sample;
    ++idx;
  }
  ASSERT_EQ(idx, transformds.size());
}

TEST(DatasetTest, PrefetchDataset) {
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  Dataset::TransformFunction scaleAndAdd = [](const af::array& a) {
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return af::sin(a) + 1.0;
  };
  auto transformDs = std::make_shared<TransformDataset>(
      tensords, std::vector<Dataset::TransformFunction>{scaleAndAdd});

  auto start = std::chrono::high_resolution_clock::now();
  for (auto& sample : *transformDs) {
    (void)sample;
  }
  auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ASSERT_NEAR(dur.count(), transformDs->size(), transformDs->size() / 5);

  // Test correctness of PrefetchDataset

  int64_t numthreads = 4;
  auto prefetchDs =
      std::make_shared<PrefetchDataset>(transformDs, numthreads, numthreads);
  for (int i = 0; i < transformDs->size(); ++i) {
    auto sample1 = transformDs->get(i);
    auto sample2 = prefetchDs->get(i);
    ASSERT_EQ(sample1.size(), sample2.size());
    for (int j = 0; j < sample1.size(); ++j) {
      ASSERT_TRUE(allClose(sample1[j], sample2[j]));
    }
  }

  // Test performance of PrefetchDataset
  start = std::chrono::high_resolution_clock::now();
  for (auto& sample : *prefetchDs) {
    (void)sample;
  }
  dur = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::high_resolution_clock::now() - start);
  ASSERT_NEAR(
      dur.count(),
      transformDs->size() / numthreads,
      transformDs->size() / numthreads / 5);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
