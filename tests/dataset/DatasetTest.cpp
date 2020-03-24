/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <thread>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/dataset/datasets.h"

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
  ResampleDataset resampleds(tensords, permfn);

  // Check `size` method
  ASSERT_EQ(resampleds.size(), 300);

  auto ff1 = resampleds.get(10);
  ASSERT_TRUE(allClose(ff1[0], tensormap[0](af::span, af::span, 15)));
  ASSERT_FALSE(allClose(ff1[0], tensormap[0](af::span, af::span, 10)));

  resampleds.resample({3, 3, 3, 4, 5});
  ASSERT_EQ(resampleds.size(), 5);

  auto ff2 = resampleds.get(1);
  ASSERT_TRUE(allClose(ff2[0], tensormap[0](af::span, af::span, 3)));
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

TEST(DatasetTest, FileBlobDataset) {
  std::vector<std::vector<af::array>> data;

  auto fillup = [&data](FileBlobDataset& blob) {
    for (int64_t i = 0; i < 20; i++) {
      std::vector<af::array> sample;
      for (int64_t j = 0; j < i % 4; j++) {
        af::array tensor;
        if (j % 2 == 0) {
          tensor = af::randu(100, 3, 100);
        } else {
          tensor = af::randu(100, 200);
        }
        sample.push_back(tensor);
      }
      data.push_back(sample);
      blob.add(sample);
    }
    blob.flush();
  };

  auto check = [&data](FileBlobDataset& blob) {
    ASSERT_EQ(data.size(), blob.size());
    for (int64_t i = 0; i < blob.size(); i++) {
      auto blob_sample = blob.get(i);
      auto dat_sample = data.at(i);
      ASSERT_EQ(dat_sample.size(), blob_sample.size());
      for (int64_t j = 0; j < blob_sample.size(); j++) {
        ASSERT_TRUE(
            af::norm(
                af::flat(dat_sample.at(j)) - af::flat(blob_sample.at(j))) <=
            1e-05);
      }
    }
  };

  // check read-write capabilities
  {
    FileBlobDataset blob("/tmp/data.blob", true, true);
    fillup(blob);
    check(blob);
    fillup(blob);
    check(blob);

    blob.writeIndex();
    fillup(blob);
    check(blob);

    blob.writeIndex();
    check(blob);

    FileBlobDataset blobcopy("/tmp/data-copy.blob", true, true);
    blobcopy.add(blob);
    blobcopy.add(blob, 1048576);
    auto datadup = data;
    data.insert(data.end(), datadup.begin(), datadup.end());
    blobcopy.writeIndex();
    check(blobcopy);
    data = datadup;
    check(blob);

    // check hostTransform
    for (auto& vec : data) {
      if (vec.size() > 0) {
        vec[0] += 1;
      }
    }
    blob.setHostTransform(
        0, [](void* ptr, af::dim4 size, af::dtype /* type */) {
          float* ptr_f = (float*)ptr;
          for (int64_t i = 0; i < size.elements(); i++) {
            ptr_f[i] += 1;
          }
          return af::array(size, ptr_f);
        });
    check(blob);
    for (auto& vec : data) {
      if (vec.size() > 0) {
        vec[0] -= 1;
      }
    }
  }

  // check everything is correct after re-opening
  {
    FileBlobDataset blob("/tmp/data.blob");
    check(blob);
  }

  // multi-threaded read
  {
    std::vector<std::vector<af::array>> thdata(data.size());
    auto blob = std::make_shared<FileBlobDataset>("/tmp/data.blob");
    std::vector<std::thread> workers;
    const int nworker = 4;
    int nperworker = data.size() / nworker;
    for (int i = 0; i < nworker; i++) {
      auto device = af::getDevice();
      workers.push_back(std::thread([i, blob, nperworker, device, &thdata]() {
        af::setDevice(device);
        for (int j = 0; j < nperworker; j++) {
          thdata[i * nperworker + j] = blob->get(i * nperworker + j);
        }
      }));
    }
    for (int i = 0; i < nworker; i++) {
      workers[i].join();
    }
    ASSERT_EQ(data.size(), thdata.size());
    for (int64_t i = 0; i < data.size(); i++) {
      auto thdata_sample = thdata.at(i);
      auto data_sample = data.at(i);
      ASSERT_EQ(data_sample.size(), thdata_sample.size());
      for (int64_t j = 0; j < thdata_sample.size(); j++) {
        ASSERT_TRUE(thdata_sample.at(j).dims() == data_sample.at(j).dims());
        ASSERT_TRUE(
            af::norm(
                af::flat(data_sample.at(j)) - af::flat(thdata_sample.at(j))) <=
            1e-05);
      }
    }
  }

  // multi-threaded write
  {
    // add an index
    for (int i = 0; i < data.size(); i++) {
      data[i].push_back(af::constant(i, 1, f32));
    }
    {
      auto blob =
          std::make_shared<FileBlobDataset>("/tmp/data.blob", true, true);
      std::vector<std::thread> workers;
      const int nworker = 10;
      int nperworker = data.size() / nworker;
      auto device = af::getDevice();
      for (int i = 0; i < nworker; i++) {
        workers.push_back(std::thread([i, blob, nperworker, device, &data]() {
          af::setDevice(device);
          for (int j = 0; j < nperworker; j++) {
            blob->add(data[i * nperworker + j]);
          }
        }));
      }
      for (int i = 0; i < nworker; i++) {
        workers[i].join();
      }
      blob->writeIndex();
    }
    {
      auto blob = std::make_shared<FileBlobDataset>("/tmp/data.blob");
      ASSERT_EQ(data.size(), blob->size());
      for (int64_t i = 0; i < data.size(); i++) {
        auto blob_sample = blob->get(i);
        auto idx = (int)blob_sample.back().scalar<float>();
        ASSERT_TRUE(idx >= 0 && idx < data.size());
        auto data_sample = data.at(idx);
        ASSERT_EQ(data_sample.size(), blob_sample.size());
        for (int64_t j = 0; j < blob_sample.size(); j++) {
          ASSERT_TRUE(data_sample.at(j).dims() == blob_sample.at(j).dims());
          ASSERT_TRUE(
              af::norm(
                  af::flat(data_sample.at(j)) - af::flat(blob_sample.at(j))) <=
              1e-05);
        }
      }
    }
  }
}

TEST(DatasetTest, PrefetchDatasetCorrectness) {
  std::vector<af::array> tensormap = {af::randu(100, 200, 300)};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  Dataset::TransformFunction scaleAndAdd = [](const af::array& a) {
    return af::cos(a) + 10.0;
  };

  auto transformDs = std::make_shared<TransformDataset>(
      tensords, std::vector<Dataset::TransformFunction>{scaleAndAdd});

  auto prefetchDs = std::make_shared<PrefetchDataset>(transformDs, 2, 2);
  for (int i = 0; i < transformDs->size(); ++i) {
    auto sample1 = transformDs->get(i);
    auto sample2 = prefetchDs->get(i);
    ASSERT_EQ(sample1.size(), sample2.size());
    for (int j = 0; j < sample1.size(); ++j) {
      ASSERT_TRUE(allClose(sample1[j], sample2[j]));
    }
  }
}

TEST(DatasetTest, DISABLED_PrefetchDatasetPerformance) {
  // Flaky test. Disabled for now.
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

  int64_t numthreads = 4;
  auto prefetchDs =
      std::make_shared<PrefetchDataset>(transformDs, numthreads, numthreads);

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
