#pragma once

#include "flashlight/dataset/datasets.h"

namespace fl {
namespace cv {
namespace dataset {

  /**
   * Creates a `DistributedDataset`, which functionally, shuffles the input data,
   * partitions it amongst the workers, prefetches it, and finally batches it.
   * @param[in] base, base dataset to load samples from
   * @param[in] world_rank the `rank` of this process in distributed training
   * @param[in] world_size number of all distributed workers
   * @param[in] num_threads number of threads used for prefetching
   * @param[in] prefetch_size max number of samples to prefetch
   */
class DistributedDataset : public Dataset {
 public:
  DistributedDataset(
      std::shared_ptr<Dataset> base,
      int64_t world_rank,
      int64_t world_size,
      int64_t batch_size,
      int64_t num_threads,
      int64_t prefetch_size);

  std::vector<af::array> get(const int64_t idx) const override;

  void resample();

  int64_t size() const override;

 private:
  std::shared_ptr<Dataset> ds_;
  std::shared_ptr<ShuffleDataset> shuffle_;
};

} // end namespace dataset
} // end namespace cv
} // end namespace fl
