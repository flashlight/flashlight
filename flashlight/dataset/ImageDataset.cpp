#include <arrayfire.h>
#include <stdlib.h>

#include "flashlight/dataset/Dataset.h"
#include "flashlight/dataset/ImageDataset.h"
#include "flashlight/dataset/MergeDataset.h"
#include "flashlight/dataset/TransformDataset.h"


namespace {

/*
 * Generic class for loading data from a vector of type T into an vector of
 * arrayfire arrays
 */
template <typename T>
class Loader : public fl::Dataset {

public:
 using LoadFunc = std::function<af::array(const T&)>;

 Loader(const std::vector<T>& list, LoadFunc loadfn)
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

}

namespace fl {


ImageDataset::ImageDataset(
    std::vector<std::string> filepaths,
    std::vector<uint64_t> labels,
    std::vector<TransformFunction>& transformfns
 ) {
  auto images = std::make_shared<Loader<std::string>>(
      filepaths, [](const std::string& filepath) {
        return af::loadimage(filepath.c_str(), true);
      });

  auto transformed = std::make_shared<TransformDataset>(
      TransformDataset(images, {compose(transformfns)})
  );

  auto targets = std::make_shared<Loader<uint64_t>>(
      labels,
      [](const uint64_t x) { return af::constant(x, 1, 1, 1, 1, u64); });
  ds_ = std::make_shared<MergeDataset>(MergeDataset({transformed, targets}));
}

std::vector<af::array> ImageDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

int64_t ImageDataset::size() const {
  return ds_->size();
}

Dataset::TransformFunction ImageDataset::resizeTransform(const uint64_t resize) {
  return [resize](const af::array& in) {
    return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
  };
}

Dataset::TransformFunction ImageDataset::compose(
    std::vector<TransformFunction>& transformfns) {
  return [transformfns](const af::array& in) {
    af::array out = in;
    for(auto fn: transformfns) {
      out = fn(out);
    }
    return out;
  };
}

Dataset::TransformFunction ImageDataset::horizontalFlipTransform(
    const float p) {
  return [p](const af::array& in) {
    af::array out = in;
    if (float(rand()) / float(RAND_MAX) > p) {
      const uint64_t w = in.dims(0);
      out = out(af::seq(w - 1, 0, -1), af::span, af::span, af::span);
    }
    return out;
  };
};

Dataset::TransformFunction ImageDataset::randomCropTransform(
    const int th,
    const int tw) {
  return [th, tw](const af::array& in) {
    af::array out = in;
    const uint64_t w = in.dims(0);
    const uint64_t h = in.dims(1);
    if (w <= tw || h <= th) {
      out = out(af::span, af::span, af::span, af::span);
    } else {
      int x = rand() % (w - tw);
      int y = rand() % (h - th);
      out = out(
          af::seq(x, x + tw - 1), af::seq(y, y + th - 1), af::span, af::span);
    }
    return out;
  };
};

Dataset::TransformFunction ImageDataset::normalizeImage(
    const std::vector<float>& mean_,
    const std::vector<float>& std_) {
  const af::array mean(1, 1, 3, 1, mean_.data());
  const af::array std(1, 1, 3, 1, std_.data());
  return [mean, std](const af::array& in) {
    auto out = in / 255.0f;
    out = af::batchFunc(out, mean, af::operator-);
    out = af::batchFunc(out, std, af::operator/);
    return out;
  };
};

} // namespace fl
