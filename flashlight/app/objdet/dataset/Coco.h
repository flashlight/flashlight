#pragma once
#include <gflags/gflags.h>

#include "flashlight/ext/image/fl/dataset/Jpeg.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/fl/dataset/datasets.h"

#include <iostream>

DECLARE_bool(onesize);
DECLARE_bool(equal_aspect);

namespace fl {
namespace app {
namespace object_detection {

struct CocoData {
  af::array images;
  af::array masks;
  af::array imageSizes;
  af::array imageIds;
  std::vector<af::array> target_boxes;
  std::vector<af::array> target_labels;
};

template<class T>
using BatchTransformFunction = std::function<
T(const std::vector<std::vector<af::array>>&)>;

template<typename T>
class BatchTransformDataset {

public:
  BatchTransformDataset(
    std::shared_ptr<const Dataset> dataset,
    int64_t batchsize,
    BatchDatasetPolicy policy /* = BatchDatasetPolicy::INCLUDE_LAST */,
    BatchTransformFunction<T> batchFn)
   : dataset_(dataset),
      batchSize_(batchsize),
      batchPolicy_(policy),
      batchFn_(batchFn) {
    if (!dataset_) {
      throw std::invalid_argument("dataset to be batched is null");
    }
    if (batchSize_ <= 0) {
      throw std::invalid_argument("invalid batch size");
    }
    preBatchSize_ = dataset_->size();
    switch (batchPolicy_) {
      case BatchDatasetPolicy::INCLUDE_LAST:
        size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      case BatchDatasetPolicy::SKIP_LAST:
        size_ = floor(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      case BatchDatasetPolicy::DIVISIBLE_ONLY:
        if (size_ % batchSize_ != 0) {
          throw std::invalid_argument(
              "dataset is not evenly divisible into batches");
        }
        size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      default:
        throw std::invalid_argument("unknown BatchDatasetPolicy");
    }
  }

  ~BatchTransformDataset() {};

  T get(const int64_t idx) {
    if (!(idx >= 0 && idx < size())) {
      throw std::out_of_range("Dataset idx out of range");
    }
    //dataset_->checkIndexBounds(idx);
    std::vector<std::vector<af::array>> buffer;

    int64_t start = batchSize_ * idx;
    int64_t end = std::min(start + batchSize_, preBatchSize_);

    for (int64_t batchidx = start; batchidx < end; ++batchidx) {
      auto fds = dataset_->get(batchidx);
      buffer.emplace_back(fds);
    }
    return batchFn_(buffer);
  }

  int64_t size() const {
    return size_;
  }

private:
  std::shared_ptr<const Dataset> dataset_;
  int64_t batchSize_;
  BatchDatasetPolicy batchPolicy_;
  BatchTransformFunction<T> batchFn_;

  int64_t preBatchSize_; // Size of the dataset before batching
  int64_t size_;

};

class CocoDataset {

public:

  CocoDataset(
      const std::string& list_file,
      std::vector<ext::image::ImageTransform>& transformfns,
      int world_rank,
      int world_size,
      int batch_size,
      int num_threads,
      int prefetch_size,
      bool val
      ); 

  std::shared_ptr<Dataset> getLabels(std::string list_file);

  std::shared_ptr<Dataset> getImages(std::string list_file,
      std::vector<ext::image::ImageTransform>& transformfns);

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

//std::shared_ptr<CocoDataset> coco(
    //const std::string& list_file,
    //std::vector<ImageTransform>& transformfns,
    //int batch_size);

//af::array cxcywh_to_xyxy(const af::array& bboxes);

//af::array xyxy_to_cxcywh(const af::array& bboxes);

//af::array xywh_to_cxcywh(const af::array& bboxes);

//af::array box_iou(const af::array& bboxes1, const af::array& bboxes2);

} // namespace dataset
} // namespace cv
} // namespace flashlight
