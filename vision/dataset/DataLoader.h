#pragma once

#include "flashlight/dataset/datasets.h"

namespace fl {
namespace cv {
namespace dataset {
/*
 * Small generic utility class for loading data from a vector of type T into an
 * vector of arrayfire arrays
 */
template <typename T>
class DataLoader : public fl::Dataset {

public:
 using LoadFunc = std::function<af::array(const T&)>;

 DataLoader(const std::vector<T>& list, LoadFunc loadfn)
     : list_(list), loadfn_(loadfn) {}

 std::vector<af::array> get(const int64_t idx) const override {
   return {loadfn_(list_[idx])};
  }

  int64_t size() const override {
    return list_.size();
  }

  private:
  std::vector<T> list_;
  LoadFunc loadfn_;
};

} // end namespace dataset
} // end namespace cv
} // end namespace fl
