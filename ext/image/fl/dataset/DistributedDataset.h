#pragma once

#include "flashlight/dataset/datasets.h"

namespace fl {
namespace ext {
namespace image {

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

} // namespace image
} // namespace ext
} // namespace fl
