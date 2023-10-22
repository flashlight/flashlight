/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/Coco.h"

#include <algorithm>
#include <cassert>
#include <map>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/vision/dataset/BoxUtils.h"
#include "flashlight/pkg/vision/dataset/CocoTransforms.h"
#include "flashlight/pkg/vision/dataset/DistributedDataset.h"
#include "flashlight/pkg/vision/dataset/LoaderDataset.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"

namespace {

using namespace fl::pkg::vision;
using namespace fl::pkg::vision;
using namespace fl;

constexpr int kElementsPerBbox = 4;

std::pair<Tensor, Tensor> makeImageAndMaskBatch(
    const std::vector<Tensor>& data) {
  int maxW = -1;
  int maxH = -1;

  for (const auto& d : data) {
    int w = d.dim(0);
    int h = d.dim(1);
    maxW = std::max(w, maxW);
    maxH = std::max(h, maxH);
  }

  Shape outDims = {maxW, maxH, 3, static_cast<long>(data.size())};
  Shape maskDims = {maxW, maxH, 1, static_cast<long>(data.size())};

  auto batcharr = fl::full(outDims, 0);
  auto maskarr = fl::full(maskDims, 0);

  for (long i = 0; i < data.size(); ++i) {
    Tensor sample = data[i];
    Shape dims = sample.shape();
    int w = dims[0];
    int h = dims[1];
    batcharr(fl::range(0, w), fl::range(0, h), fl::span, fl::range(i, i + 1)) =
        data[i];
    maskarr(fl::range(0, w), fl::range(0, h), fl::span, fl::range(i, i + 1)) =
        fl::full({w, h}, 1);
  }
  return std::make_pair(batcharr, maskarr);
}

// Since the bboxes and classes are variable length, we don't actually want
// to batch them together.
CocoData cocoBatchFunc(const std::vector<std::vector<Tensor>>& batches) {
  Tensor imageBatch, masks;
  std::tie(imageBatch, masks) = makeImageAndMaskBatch(batches[ImageIdx]);
  return {
      imageBatch,
      masks,
      makeBatch(batches[TargetSizeIdx]),
      makeBatch(batches[ImageIdIdx]),
      makeBatch(batches[OriginalSizeIdx]),
      batches[BboxesIdx],
      batches[ClassesIdx]};
}

int64_t getImageId(const std::string& fp) {
  const std::string slash("/");
  const std::string period(".");
  int start = fp.rfind(slash);
  int end = fp.rfind(period);
  std::string substring = fp.substr(start + 1, end - start);
  return std::stol(substring);
}

} // namespace

namespace fl::pkg::vision {

CocoDataset::CocoDataset(
    const std::string& list_file,
    int world_rank,
    int world_size,
    int batch_size,
    int num_threads,
    int prefetch_size,
    bool val) {
  // Create vector of CocoDataSample which will be loaded into arrayfire arrays
  std::vector<CocoDataSample> data;
  std::ifstream ifs(list_file);
  if (!ifs) {
    throw std::runtime_error("Could not open list file: " + list_file);
  }
  // We use tabs a deliminators between the filepath and each bbox
  // We use spaced to separate the different fields of the bbox
  const std::string delim = "\t";
  const std::string bbox_delim = " ";
  std::string line;
  while (std::getline(ifs, line)) {
    int item = line.find(delim);
    std::string filepath = line.substr(0, item);
    std::vector<float> bboxes;
    std::vector<float> classes;
    item = line.find(delim, item);
    while (item != std::string::npos) {
      int pos = item;
      int next;
      for (int i = 0; i < 4; i++) {
        next = line.find(bbox_delim, pos + 1);
        assert(next != std::string::npos);
        bboxes.emplace_back(std::stof(line.substr(pos, next - pos)));
        pos = next;
      }
      next = line.find(bbox_delim, pos + 1);
      classes.emplace_back(std::stod(line.substr(pos, next - pos)));
      item = line.find(delim, pos);
    }
    data.emplace_back(CocoDataSample{filepath, bboxes, classes});
  }
  assert(!data.empty());

  // Now define how to load the data from CocoDataSampoles in arrayfire
  std::shared_ptr<Dataset> ds = std::make_shared<LoaderDataset<CocoDataSample>>(
      data, [](const CocoDataSample& sample) {
        Tensor image = loadJpeg(sample.filepath);

        std::vector<long> targetSizes = {image.dim(1), image.dim(0)};
        Tensor targetSize = Tensor::fromVector(targetSizes);
        Tensor imageId =
            fl::full({getImageId(sample.filepath)}, 1, fl::dtype::s64);

        const int num_elements = sample.bboxes.size();
        const int num_bboxes = num_elements / kElementsPerBbox;
        Tensor bboxes, classes;
        if (num_bboxes > 0) {
          bboxes =
              Tensor::fromVector({kElementsPerBbox, num_bboxes}, sample.bboxes);
          classes = Tensor::fromVector({1, num_bboxes}, sample.classes);
        } else {
          // Arrayfire doesn't allow you to create 0 length dimension on
          // anything other than the first dimension so we need this switch
          bboxes = Tensor();
          classes = Tensor();
        }
        // image, size, imageId, original_size
        return std::vector<Tensor>{
            image, targetSize, imageId, targetSize, bboxes, classes};
      });

  const int maxSize = 1333;
  if (val) {
    ds =
        std::make_shared<TransformAllDataset>(ds, randomResize({800}, maxSize));
  } else {
    std::vector<int> scales = {
        480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800};
    TransformAllFunction trainTransform = compose(
        {randomHorizontalFlip(0.5),
         randomSelect(
             {randomResize(scales, maxSize),
              compose(
                  {randomResize({400, 500, 600}, -1),
                   randomSizeCrop(384, 600),
                   randomResize(scales, 1333)})})});

    ds = std::make_shared<TransformAllDataset>(ds, trainTransform);
  }

  ds = std::make_shared<TransformAllDataset>(ds, Normalize());

  // Skip shuffling if doing eval.
  if (!val) {
    shuffled_ = std::make_shared<ShuffleDataset>(ds);
    ds = shuffled_;
  }
  auto permfn = [world_size, world_rank](int64_t idx) {
    return (idx * world_size) + world_rank;
  };

  ds = std::make_shared<ResampleDataset>(ds, permfn, ds->size() / world_size);
  ds = std::make_shared<PrefetchDataset>(ds, num_threads, prefetch_size);
  batched_ = std::make_shared<BatchTransformDataset<CocoData>>(
      ds, batch_size, BatchDatasetPolicy::SKIP_LAST, cocoBatchFunc);
}

void CocoDataset::resample() {
  if (shuffled_) {
    shuffled_->resample();
  }
}
int64_t CocoDataset::size() const {
  return batched_->size();
}

CocoData CocoDataset::get(const uint64_t idx) {
  return batched_->get(idx);
}

} // namespace fl
