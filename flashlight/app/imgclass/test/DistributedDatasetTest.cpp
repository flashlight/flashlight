/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "flashlight/app/imgclass/dataset/Imagenet.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/fl/common/Init.h"
#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"

using namespace fl;
using namespace fl::ext::image;
using namespace fl::app::imgclass;

const std::vector<float> mean = {0.485, 0.456, 0.406};
const std::vector<float> stdv = {0.229, 0.224, 0.225};
const int randomResizeMax = 480;
const int randomResizeMin = 256;
const int randomCropSize = 224;
const float horizontalFlipProb = 0.5f;

const float pRandomerase = 0.25;
const float pRandomeaug = 0.5;
const int nRandomeaug = 2;

const int64_t nSamples = 50000;
const int64_t batchSizePerGpu = 50;
const int64_t prefetchSize = batchSizePerGpu * 10;

const std::string dataDir = "/datasets01/imagenet_full_size/061417";
const std::string labelPath = lib::pathsConcat(dataDir, "labels.txt");
const std::string trainList = lib::pathsConcat(dataDir, "val");

// TEST(DistributedDatasetTest, Loading) {
//   const ImageTransform trainTransforms = compose({
//       fl::ext::image::randomResizeTransform(randomResizeMin, randomResizeMax),
//       fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
//       fl::ext::image::randomAugmentationTransform(pRandomeaug, nRandomeaug),
//       fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb),
//       fl::ext::image::normalizeImage(mean, stdv),
//       fl::ext::image::randomEraseTransform(pRandomerase)
//       // end
//   });
//   auto labelMap = getImagenetLabels(labelPath);
//   auto trainDataset = fl::ext::image::DistributedDataset(
//       imagenetDataset(trainList, labelMap, {trainTransforms}),
//       0, // worldRank,s
//       1, // worldSize,
//       batchSizePerGpu,
//       1, // FLAGS_train_n_repeatedaug,
//       10, // FLAGS_data_prefetch_thread,
//       prefetchSize,
//       fl::BatchDatasetPolicy::INCLUDE_LAST);

//   ASSERT_EQ(trainDataset.size(), nSamples / batchSizePerGpu);

//   auto sample = trainDataset.get(0)[kImagenetInputIdx];
//   ASSERT_EQ(sample.dims(0), 224);
//   ASSERT_EQ(sample.dims(1), 224);
//   ASSERT_EQ(sample.dims(2), 3);
//   ASSERT_EQ(sample.dims(3), batchSizePerGpu);

//   auto target = trainDataset.get(0)[kImagenetTargetIdx];
//   ASSERT_EQ(target.dims(0), batchSizePerGpu);
// }

// TEST(DistributedDatasetTest, Shuffle) {
//   const ImageTransform trainTransforms = compose({
//       fl::ext::image::randomResizeTransform(randomResizeMin, randomResizeMax),
//       fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
//       fl::ext::image::randomAugmentationTransform(pRandomeaug, nRandomeaug),
//       fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb),
//       fl::ext::image::normalizeImage(mean, stdv),
//       fl::ext::image::randomEraseTransform(pRandomerase)
//       // end
//   });
//   auto labelMap = getImagenetLabels(labelPath);
//   auto trainDataset = fl::ext::image::DistributedDataset(
//       imagenetDataset(trainList, labelMap, {trainTransforms}),
//       0, // worldRank,
//       1, // worldSize,
//       batchSizePerGpu,
//       3, // FLAGS_train_n_repeatedaug,
//       10, // FLAGS_data_prefetch_thread,
//       prefetchSize,
//       fl::BatchDatasetPolicy::INCLUDE_LAST);

//   auto target1 = trainDataset.get(0)[kImagenetTargetIdx];
//   trainDataset.resample(4399);
//   auto target2 = trainDataset.get(0)[kImagenetTargetIdx];
//   ASSERT_TRUE(!af::allTrue<bool>(target1 == target2));
// }

TEST(DistributedDatasetTest, RepeatedAug) {
  const ImageTransform trainTransforms = compose({
      fl::ext::image::randomResizeTransform(randomResizeMin, randomResizeMax),
      fl::ext::image::randomCropTransform(randomCropSize, randomCropSize),
      fl::ext::image::randomAugmentationTransform(pRandomeaug, nRandomeaug),
      fl::ext::image::randomHorizontalFlipTransform(horizontalFlipProb),
      fl::ext::image::normalizeImage(mean, stdv),
      fl::ext::image::randomEraseTransform(pRandomerase)
      // end
  });
  auto labelMap = getImagenetLabels(labelPath);
  auto imnet = imagenetDataset(trainList, labelMap, {trainTransforms});

  int worldSize = 8;
  std::vector<fl::ext::image::DistributedDataset> datasets;
  for (int i = 0; i < worldSize; i++) {
    datasets.emplace_back(
        imagenetDataset(trainList, labelMap, {trainTransforms}),
        i, // worldRank,
        worldSize, // worldSize,
        10, // batchSizePerGpu,
        3, // FLAGS_train_n_repeatedaug,
        10, // FLAGS_data_prefetch_thread,
        prefetchSize,
        fl::BatchDatasetPolicy::INCLUDE_LAST);
  }

  for (int e = 0; e < 10; e++) {
    for (int i = 0; i < worldSize; i++) {
      datasets[i].resample(e);
    }
  }

  for (int i = 0; i < worldSize; i++) {
    auto target = datasets[i].get(1)[kImagenetTargetIdx];
    af_print(target);
  }

  // af::array target1;
  // for (int i = 0; i < 10; i++) {
  //   target1 = trainDataset1.get(i)[kImagenetTargetIdx];
  //   auto target2 = trainDataset2.get(i)[kImagenetTargetIdx];
  //   auto target3 = trainDataset3.get(i)[kImagenetTargetIdx];
  //   // af_print(target1);
  //   // af_print(target2);
  //   // af_print(target3);
  //   ASSERT_TRUE(af::allTrue<bool>(target1 == target2));
  //   ASSERT_TRUE(af::allTrue<bool>(target1 == target3));
  // }
  // trainDataset1.resample(4399);
  // trainDataset2.resample(4399);
  // trainDataset3.resample(4399);
  // auto newTarget1 = trainDataset1.get(9)[kImagenetTargetIdx];
  // auto newTarget2 = trainDataset2.get(9)[kImagenetTargetIdx];
  // auto newTarget3 = trainDataset3.get(9)[kImagenetTargetIdx];
  // // af_print(newTarget1);
  // ASSERT_TRUE(!af::allTrue<bool>(target1 == newTarget1));
  // ASSERT_TRUE(af::allTrue<bool>(newTarget1 == newTarget2));
  // ASSERT_TRUE(af::allTrue<bool>(newTarget1 == newTarget3));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  fl::init();

  return RUN_ALL_TESTS();
}
