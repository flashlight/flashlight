#include <arrayfire.h>
#include <stdlib.h>
#include <iostream>

#include "flashlight/dataset/Dataset.h"
#include "flashlight/dataset/ImageDataset.h"
#include "flashlight/dataset/MergeDataset.h"
#include "flashlight/dataset/TransformDataset.h"
#define STB_IMAGE_IMPLEMENTATION
#include "flashlight/dataset/stb_image.h"


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
	int width, height, channels;
	int desired_no_channels = 3;
	unsigned char *img = stbi_load(
      filepath.c_str(), 
      &width, 
      &height, 
      &channels, 
      desired_no_channels
  );
	if (img) {
		af::array result = af::array(width, height, desired_no_channels, img).as(f32);
		stbi_image_free(img);
		return result;
	} else {
		std::cout << " channels " << channels << std::endl;
		std::cout << "filepath:  " << filepath << std::endl;
		return af::constant(0, 244, 244, 3);
	}
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
    int th, tw;
    int w = in.dims(0);
    int h = in.dims(1);
    if (h > w) {
      th = (resize * h) / w;
      tw = resize;
    } else {
      th = resize;
      tw = (resize * w) / h;
    }
    return af::resize(in, tw, th, AF_INTERP_BILINEAR);
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
Dataset::TransformFunction ImageDataset::centerCrop(
    const int size) {
  return [size](const af::array& in) {
    const int w = in.dims(0);
    const int h = in.dims(1);
    const int cropTop = (h - size) / 2;
    const int cropLeft = (w - size) / 2;
    const af::array result = in(
        af::seq(cropLeft, cropLeft + size - 1),//            
        af::seq(cropTop, cropTop + size - 1),//          
        af::span);
    return result;
  };
};

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
  int x = rand() % (w - tw + 1);
	int y = rand() % (h - th + 1);
	out = out(
  af::seq(x, x + tw - 1), af::seq(y, y + th - 1), af::span, af::span);
  std::cout << out.dims() << std::endl;
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
