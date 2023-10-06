/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <thread>

#include <gtest/gtest.h>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"
#include "flashlight/fl/tensor/Random.h"

using namespace fl;

TEST(DatasetTest, TensorDataset) {
  std::vector<Tensor> tensormap = {
      fl::rand({100, 200, 300}), fl::rand({150, 300})};
  TensorDataset tensords(tensormap);

  // Check `size` method
  ASSERT_EQ(tensords.size(), 300);

  // Values using `get` method
  auto ff1 = tensords.get(10);
  ASSERT_EQ(ff1.size(), 2);
  ASSERT_TRUE(allClose(ff1[0], tensormap[0](fl::span, fl::span, 10)));
  ASSERT_TRUE(allClose(ff1[1], tensormap[1](fl::span, 10)));
}

TEST(DatasetTest, TranformDataset) {
  // first create a tensor dataset
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  auto scaleAndAdd = [](const Tensor& a) { return fl::sin(a) + 1.0; };
  TransformDataset transformds(tensords, {scaleAndAdd});

  // Check `size` method
  ASSERT_TRUE(transformds.size() == 300);

  // Values using `get` method
  auto ff1 = transformds.get(10);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], fl::sin(tensormap[0](fl::span, fl::span, 10)) + 1.0));
}

TEST(DatasetTest, BatchDataset) {
  // first create a tensor dataset
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  BatchDataset batchds(tensords, 7, BatchDatasetPolicy::INCLUDE_LAST);

  // Check `size` method
  ASSERT_EQ(batchds.size(), 43);

  // Values using `get` method
  auto ff1 = batchds.get(42);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], tensormap[0](fl::span, fl::span, fl::range(294, 300))));

  ff1 = batchds.get(10);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], tensormap[0](fl::span, fl::span, fl::range(70, 77))));
}

TEST(DatasetTest, DynamicBatchDataset) {
  // first create a tensor dataset
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);
  std::vector<int64_t> bSzs = {20, 50, 20, 30, 10, 50, 20, 35, 15, 50};
  BatchDataset batchds(tensords, bSzs);

  // Check `size` method
  ASSERT_EQ(batchds.size(), bSzs.size());

  // Values using `get` method
  auto ff1 = batchds.get(0);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], tensormap[0](fl::span, fl::span, fl::range(0, 20))));

  ff1 = batchds.get(3);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(
      allClose(ff1[0], tensormap[0](fl::span, fl::span, fl::range(90, 120))));
}

TEST(DatasetTest, ShuffleDataset) {
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);
  ShuffleDataset shuffleds(tensords);

  // Check `size` method
  ASSERT_EQ(shuffleds.size(), 300);

  // Values using `get` method
  auto ff1 = shuffleds.get(10);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_FALSE(allClose(ff1[0], tensormap[0](fl::span, fl::span, 10)));

  // Same seed produces same order and vice-versa
  ShuffleDataset shuffleds2(tensords, 2);
  ShuffleDataset shuffleds3(tensords, 2);
  ShuffleDataset shuffleds4(tensords, 3);
  auto ff2 = shuffleds2.get(10);
  auto ff3 = shuffleds3.get(10);
  auto ff4 = shuffleds4.get(10);
  ASSERT_EQ(ff2.size(), 1);
  ASSERT_EQ(ff3.size(), 1);
  ASSERT_EQ(ff4.size(), 1);
  ASSERT_TRUE(allClose(ff2[0], ff3[0]));
  ASSERT_FALSE(allClose(ff2[0], ff4[0]));
}

TEST(DatasetTest, ResampleDataset) {
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);
  auto permfn = [](int64_t n) { return (n + 5) % 300; };
  ResampleDataset resampleds(tensords, permfn);

  // Check `size` method
  ASSERT_EQ(resampleds.size(), 300);

  auto ff1 = resampleds.get(10);
  ASSERT_TRUE(allClose(ff1[0], tensormap[0](fl::span, fl::span, 15)));
  ASSERT_FALSE(allClose(ff1[0], tensormap[0](fl::span, fl::span, 10)));

  resampleds.resample({3, 3, 3, 4, 5});
  ASSERT_EQ(resampleds.size(), 5);

  auto ff2 = resampleds.get(1);
  ASSERT_TRUE(allClose(ff2[0], tensormap[0](fl::span, fl::span, 3)));
}

TEST(DatasetTest, SpanDataset) {
  std::vector<Tensor> tensormap = {
      fl::rand({100, 200, 300}), fl::rand({150, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  SpanDataset frontspands(tensords, 0, 13);
  SpanDataset backspands(tensords, 13);

  // Check `size` method
  ASSERT_EQ(frontspands.size(), 13);
  ASSERT_EQ(backspands.size(), 287);

  // Values using `get` method
  auto ff1 = frontspands.get(10);
  ASSERT_EQ(ff1.size(), 2);
  ASSERT_TRUE(allClose(ff1[0], tensormap[0](fl::span, fl::span, 10)));
  ASSERT_TRUE(allClose(ff1[1], tensormap[1](fl::span, 10)));
  auto ff2 = backspands.get(10);
  ASSERT_EQ(ff2.size(), 2);
  ASSERT_TRUE(allClose(ff2[0], tensormap[0](fl::span, fl::span, 13 + 10)));
  ASSERT_TRUE(allClose(ff2[1], tensormap[1](fl::span, 13 + 10)));
}

TEST(DatasetTest, ConcatDataset) {
  auto tensor1 = fl::rand({100, 200, 100});
  auto tensor2 = fl::rand({100, 200, 200});
  std::vector<Tensor> tensormap1 = {tensor1};
  auto tensords1 = std::make_shared<TensorDataset>(tensormap1);
  std::vector<Tensor> tensormap2 = {tensor2};
  auto tensords2 = std::make_shared<TensorDataset>(tensormap2);
  ConcatDataset concatds({tensords1, tensords2});

  // Check `size` method
  ASSERT_TRUE(concatds.size() == 300);

  auto ff1 = concatds.get(100);
  ASSERT_EQ(ff1.size(), 1);
  ASSERT_TRUE(allClose(ff1[0], tensor2(fl::span, fl::span, 0)));
  ff1 = concatds.get(299);
  ASSERT_TRUE(allClose(ff1[0], tensor2(fl::span, fl::span, 199)));
  ff1 = concatds.get(0);
  ASSERT_TRUE(allClose(ff1[0], tensor1(fl::span, fl::span, 0)));
  ff1 = concatds.get(10);
  ASSERT_TRUE(allClose(ff1[0], tensor1(fl::span, fl::span, 10)));
}

TEST(DatasetTest, DatasetIterator) {
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  auto scaleAndAdd = [](const Tensor& a) { return fl::sin(a) + 1.0; };
  TransformDataset transformds(tensords, {scaleAndAdd});

  int idx = 0;
  for (auto& sample : transformds) {
    (void)sample;
    ++idx;
  }
  ASSERT_EQ(idx, transformds.size());
}

TEST(DatasetTest, FileBlobDataset) {
  std::vector<std::vector<Tensor>> data;

  auto fillup = [&data](FileBlobDataset& blob) {
    for (int64_t i = 0; i < 20; i++) {
      std::vector<Tensor> sample;
      for (int64_t j = 0; j < i % 4; j++) {
        Tensor tensor;
        if (j % 2 == 0) {
          tensor = fl::rand({100, 3, 100});
        } else {
          tensor = fl::rand({100, 200});
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
      auto blobSample = blob.get(i);
      auto datSample = data.at(i);
      ASSERT_EQ(datSample.size(), blobSample.size());
      for (int64_t j = 0; j < blobSample.size(); j++) {
        ASSERT_TRUE(
            fl::norm(datSample.at(j).flatten() - blobSample.at(j).flatten())
                .scalar<float>() <= 1e-05);
      }
    }
  };

  // check read-write capabilities
  {
    FileBlobDataset blob(fs::temp_directory_path() / "data.blob", true, true);
    fillup(blob);
    check(blob);
    fillup(blob);
    check(blob);

    blob.writeIndex();
    fillup(blob);
    check(blob);

    blob.writeIndex();
    check(blob);

    FileBlobDataset blobcopy(
        fs::temp_directory_path() / "data-copy.blob", true, true);
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
      if (!vec.empty()) {
        vec[0] += 1;
      }
    }
    blob.setHostTransform(
        0, [](void* ptr, fl::Shape size, fl::dtype /* type */) {
          float* ptrFl = (float*)ptr;
          for (int64_t i = 0; i < size.elements(); i++) {
            ptrFl[i] += 1;
          }
          return Tensor::fromBuffer(size, ptrFl, MemoryLocation::Host);
        });
    check(blob);
    for (auto& vec : data) {
      if (!vec.empty()) {
        vec[0] -= 1;
      }
    }
  }

  // check tensor dim constraints
  {
    FileBlobDataset blob(
        fs::temp_directory_path() / "max_size.blob", true, true);
    ASSERT_THROW(blob.add({fl::rand({4, 5, 6, 7, 8})}), std::invalid_argument);
  }

  // check everything is correct after re-opening
  {
    FileBlobDataset blob(fs::temp_directory_path() / "data.blob");
    check(blob);
  }

  // multi-threaded read
  {
    std::vector<std::vector<Tensor>> thdata(data.size());
    auto blob = std::make_shared<FileBlobDataset>(
        fs::temp_directory_path() / "data.blob");
    std::vector<std::thread> workers;
    const int nworker = 4;
    int nperworker = data.size() / nworker;
    for (int i = 0; i < nworker; i++) {
      auto device = fl::getDevice();
      workers.emplace_back([i, blob, nperworker, device, &thdata]() {
        fl::setDevice(device);
        for (int j = 0; j < nperworker; j++) {
          thdata[i * nperworker + j] = blob->get(i * nperworker + j);
        }
      });
    }
    for (int i = 0; i < nworker; i++) {
      workers[i].join();
    }
    ASSERT_EQ(data.size(), thdata.size());
    for (int64_t i = 0; i < data.size(); i++) {
      auto thdataSample = thdata.at(i);
      auto dataSample = data.at(i);
      ASSERT_EQ(dataSample.size(), thdataSample.size());
      for (int64_t j = 0; j < thdataSample.size(); j++) {
        ASSERT_TRUE(thdataSample.at(j).shape() == dataSample.at(j).shape());
        ASSERT_TRUE(
            fl::norm(dataSample.at(j).flatten() - thdataSample.at(j).flatten())
                .scalar<float>() <= 1e-05);
      }
    }
  }

  // multi-threaded write
  {
    // add an index
    for (int i = 0; i < data.size(); i++) {
      data[i].push_back(fl::full({1}, i, fl::dtype::f32));
    }
    {
      auto blob = std::make_shared<FileBlobDataset>(
          fs::temp_directory_path() / "data.blob", true, true);
      std::vector<std::thread> workers;
      const int nworker = 10;
      int nperworker = data.size() / nworker;
      auto device = fl::getDevice();
      for (int i = 0; i < nworker; i++) {
        workers.emplace_back([i, blob, nperworker, device, &data]() {
          fl::setDevice(device);
          for (int j = 0; j < nperworker; j++) {
            blob->add(data[i * nperworker + j]);
          }
        });
      }
      for (int i = 0; i < nworker; i++) {
        workers[i].join();
      }
      blob->writeIndex();
    }
    {
      auto blob = std::make_shared<FileBlobDataset>(
          fs::temp_directory_path() / "data.blob");
      ASSERT_EQ(data.size(), blob->size());
      for (int64_t i = 0; i < data.size(); i++) {
        auto blobSample = blob->get(i);
        auto idx = (int)blobSample.back().scalar<float>();
        ASSERT_TRUE(idx >= 0 && idx < data.size());
        auto dataSample = data.at(idx);
        ASSERT_EQ(dataSample.size(), blobSample.size());
        for (int64_t j = 0; j < blobSample.size(); j++) {
          ASSERT_TRUE(dataSample.at(j).shape() == blobSample.at(j).shape());
          ASSERT_TRUE(
              fl::norm(dataSample.at(j).flatten() - blobSample.at(j).flatten())
                  .scalar<float>() <= 1e-05);
        }
      }
    }
  }
}

TEST(DatasetTest, MemoryBlobDataset) {
  std::vector<std::vector<Tensor>> data;

  auto fillup = [&data](MemoryBlobDataset& blob) {
    for (int64_t i = 0; i < 20; i++) {
      std::vector<Tensor> sample;
      for (int64_t j = 0; j < i % 4; j++) {
        Tensor tensor;
        if (j % 2 == 0) {
          tensor = fl::rand({100, 3, 100});
        } else {
          tensor = fl::rand({100, 200});
        }
        sample.push_back(tensor);
      }
      data.push_back(sample);
      blob.add(sample);
    }
    blob.flush();
  };

  auto check = [&data](MemoryBlobDataset& blob) {
    ASSERT_EQ(data.size(), blob.size());
    for (int64_t i = 0; i < blob.size(); i++) {
      auto blobSample = blob.get(i);
      auto datSample = data.at(i);
      ASSERT_EQ(datSample.size(), blobSample.size());
      for (int64_t j = 0; j < blobSample.size(); j++) {
        ASSERT_TRUE(
            fl::norm(datSample.at(j).flatten() - blobSample.at(j).flatten())
                .scalar<float>() <= 1e-05);
      }
    }
  };

  // check read-write capabilities
  MemoryBlobDataset blob;
  {
    fillup(blob);
    check(blob);
    fillup(blob);
    check(blob);

    blob.writeIndex();
    fillup(blob);
    check(blob);

    blob.writeIndex();
    check(blob);

    MemoryBlobDataset blobcopy;
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
      if (!vec.empty()) {
        vec[0] += 1;
      }
    }
    blob.setHostTransform(
        0, [](void* ptr, fl::Shape size, fl::dtype /* type */) {
          float* ptrFl = (float*)ptr;
          for (int64_t i = 0; i < size.elements(); i++) {
            ptrFl[i] += 1;
          }
          return Tensor::fromBuffer(size, ptrFl, MemoryLocation::Host);
        });
    check(blob);
  }

  // multi-threaded read
  {
    std::vector<std::vector<Tensor>> thdata(data.size());
    std::vector<std::thread> workers;
    const int nworker = 4;
    int nperworker = data.size() / nworker;
    for (int i = 0; i < nworker; i++) {
      auto device = fl::getDevice();
      workers.emplace_back([i, &blob, nperworker, device, &thdata]() {
        fl::setDevice(device);
        for (int j = 0; j < nperworker; j++) {
          thdata[i * nperworker + j] = blob.get(i * nperworker + j);
        }
      });
    }
    for (int i = 0; i < nworker; i++) {
      workers[i].join();
    }
    ASSERT_EQ(data.size(), thdata.size());
    for (int64_t i = 0; i < data.size(); i++) {
      auto thdataSample = thdata.at(i);
      auto dataSample = data.at(i);
      ASSERT_EQ(dataSample.size(), thdataSample.size());
      for (int64_t j = 0; j < thdataSample.size(); j++) {
        ASSERT_TRUE(thdataSample.at(j).shape() == dataSample.at(j).shape());
        ASSERT_TRUE(
            fl::norm(dataSample.at(j).flatten() - thdataSample.at(j).flatten())
                .scalar<float>() <= 1e-05);
      }
    }
  }

  // multi-threaded write
  {
    MemoryBlobDataset wblob;
    // add an index
    for (int i = 0; i < data.size(); i++) {
      data[i].push_back(fl::full({1}, i, fl::dtype::f32));
    }
    {
      std::vector<std::thread> workers;
      const int nworker = 10;
      int nperworker = data.size() / nworker;
      auto device = fl::getDevice();
      for (int i = 0; i < nworker; i++) {
        workers.emplace_back([i, &wblob, nperworker, device, &data]() {
          fl::setDevice(device);
          for (int j = 0; j < nperworker; j++) {
            wblob.add(data[i * nperworker + j]);
          }
        });
      }
      for (int i = 0; i < nworker; i++) {
        workers[i].join();
      }
      wblob.writeIndex();
    }
    {
      ASSERT_EQ(data.size(), wblob.size());
      for (int64_t i = 0; i < data.size(); i++) {
        auto wblobSample = wblob.get(i);
        auto idx = (int)wblobSample.back().scalar<float>();
        ASSERT_TRUE(idx >= 0 && idx < data.size());
        auto dataSample = data.at(idx);
        ASSERT_EQ(dataSample.size(), wblobSample.size());
        for (int64_t j = 0; j < wblobSample.size(); j++) {
          ASSERT_TRUE(dataSample.at(j).shape() == wblobSample.at(j).shape());
          ASSERT_TRUE(
              fl::norm(dataSample.at(j).flatten() - wblobSample.at(j).flatten())
                  .scalar<float>() <= 1e-05);
        }
      }
    }
  }
}

TEST(DatasetTest, PrefetchDatasetCorrectness) {
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  Dataset::TransformFunction scaleAndAdd = [](const Tensor& a) {
    return fl::cos(a) + 10.0;
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
  std::vector<Tensor> tensormap = {fl::rand({100, 200, 300})};
  auto tensords = std::make_shared<TensorDataset>(tensormap);

  Dataset::TransformFunction scaleAndAdd = [](const Tensor& a) {
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return fl::sin(a) + 1.0;
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
  fl::init();
  return RUN_ALL_TESTS();
}
