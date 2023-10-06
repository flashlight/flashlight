/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/pkg/vision/dataset/BatchTransformDataset.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"
#include "flashlight/pkg/vision/dataset/Jpeg.h"
#include "flashlight/fl/dataset/datasets.h"

namespace fl {
namespace pkg {
namespace vision {

struct CocoDataSample {
  std::string filepath;
  std::vector<float> bboxes;
  std::vector<float> classes;
};

struct CocoData {
  Tensor images;
  Tensor masks;
  Tensor imageSizes;
  Tensor imageIds;
  Tensor originalImageSizes;
  std::vector<Tensor> target_boxes;
  std::vector<Tensor> target_labels;
};

class CocoDataset {
 public:
  CocoDataset(
      const std::string& list_file,
      int world_rank,
      int world_size,
      int batch_size,
      int num_threads,
      int prefetch_size,
      bool val);

  std::shared_ptr<Dataset> getLabels(std::string list_file);

  std::shared_ptr<Dataset> getImages(
      std::string list_file,
      std::vector<pkg::vision::ImageTransform>& transformfns);

  using iterator = detail::DatasetIterator<CocoDataset, CocoData>;

  iterator begin() {
    return iterator(this);
  }

  iterator end() {
    return iterator();
  }

  int64_t size() const;

  CocoData get(const uint64_t idx);

  void resample();

 private:
  std::shared_ptr<BatchTransformDataset<CocoData>> batched_;
  std::shared_ptr<ShuffleDataset> shuffled_;
};

} // namespace vision
} // namespace pkg
} // namespace fl
