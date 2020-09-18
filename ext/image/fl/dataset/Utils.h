#pragma once

#include "flashlight/dataset/datasets.h"
#include <glob.h>

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

  std::vector<af::array> get(const int64_t idx) const override {
    checkIndexBounds(idx);
    return ds_->get(idx);
  }

  void resample() {
    shuffle_->resample();
  }

  int64_t size() const override {
    return ds_->size();
  }

 private:
  std::shared_ptr<Dataset> ds_;
  std::shared_ptr<ShuffleDataset> shuffle_;
};

/*
 * Small generic utility class for loading data from a vector of type T into an
 * vector of arrayfire arrays
 */
template <typename T>
class Loader : public fl::Dataset {

public:
 using LoadFunc = std::function<std::vector<af::array>(const T&)>;

 Loader(const std::vector<T>& list, LoadFunc loadfn)
     : list_(list), loadfn_(loadfn) {}

 std::vector<af::array> get(const int64_t idx) const override {
   return loadfn_(list_[idx]);
  }

  int64_t size() const override {
    return list_.size();
  }

  private:
  std::vector<T> list_;
  LoadFunc loadfn_;
};

inline std::vector<std::string> glob(const std::string& pat) {
  glob_t result;
  glob(pat.c_str(), GLOB_TILDE, nullptr, &result);
  std::vector<std::string> ret;
  for (unsigned int i = 0; i < result.gl_pathc; ++i) {
    ret.push_back(std::string(result.gl_pathv[i]));
  }
  globfree(&result);
  return ret;
}

} // namespace image
} // namespace ext
} // namespace fl
