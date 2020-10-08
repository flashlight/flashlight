#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"

namespace fl {
namespace ext {
namespace image {

DistributedDataset::DistributedDataset(
    std::shared_ptr<Dataset> base,
    int64_t world_rank,
    int64_t world_size,
    int64_t batch_size,
    int64_t num_threads,
    int64_t prefetch_size) {
  shuffle_ = std::make_shared<ShuffleDataset>(base);
  auto permfn = [world_size, world_rank](int64_t idx) {
    return (idx * world_size) + world_rank;
  };
  ds_ = std::make_shared<ResampleDataset>(
    shuffle_, permfn, shuffle_->size() / world_size);
  ds_ = std::make_shared<PrefetchDataset>(ds_, num_threads, prefetch_size);
  ds_ = std::make_shared<BatchDataset>(ds_, batch_size);
}

std::vector<af::array> DistributedDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

void DistributedDataset::resample() {
  shuffle_->resample();
}

int64_t DistributedDataset::size() const {
  return ds_->size();
}

} // namespace image
} // namespace ext
} // namespace fl
