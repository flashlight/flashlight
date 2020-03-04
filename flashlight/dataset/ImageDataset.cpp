#include <thread>
#include <chrono>

#include <cudnn.h>
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
 * Small generic utility class for loading data from a vector of type T into an 
 * vector of arrayfire arrays
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

/*
 * Resizes the smallest length edge of an image to be resize while keeping 
 * the aspect ratio
 */
af::array resizeSmallest(const af::array in, const int resize) {
    const int w = in.dims(0);
    const int h = in.dims(1);
    int th, tw;
    if (h > w) {
      th = (resize * h) / w;
      tw = resize;
    } else {
      th = resize;
      tw = (resize * w) / h;
    }
    return af::resize(in, tw, th, AF_INTERP_BILINEAR);
}

/*
 * Loads a jpeg from filepath fp. Note: It will automatically convert from any
 * numnber of channels to create an array with 3 channels
 */
af::array loadJpeg(const std::string& fp) {
  af::array img;
  try {
    img = af::loadImageNative(fp.c_str());
  } catch (...){
    img = af::constant(0, 224, 244, 3);
    std::cout << "Filepath " << fp << std::endl;
  }
  if (img.dims(2) == 3) {
    return img;
  } else if (img.dims(2) == 1) {
    img = af::tile(img, 2, 3);
    return img;
  }
  /*
	int w, h, c;
  // STB image will automatically return desired_no_channels. 
  // NB: c will be the original number of channels
	int desired_no_channels = 3;
	unsigned char *img = stbi_load(fp.c_str(), &w, &h, &c, desired_no_channels);
	if (img) {
		af::array result = af::array(w, h, desired_no_channels, img).as(f32);
		stbi_image_free(img);
		return result;
	} else {
    throw std::invalid_argument("Could not load from filepath" + fp);
	}
  */
}

af::array loadLabel(const uint64_t x) {
  return af::constant(x, 1, 1, 1, 1, u64); 
}

}

namespace fl {

ImageDataset::ImageDataset(
    std::vector<std::string> filepaths,
    std::vector<uint64_t> labels,
    std::vector<TransformFunction>& transformfns
 ) {

  // Create image loader and apply transforms
  auto images = std::make_shared<Loader<std::string>>(filepaths, loadJpeg);
  // TransformDataset will apply each transform in a vector to the respective af::array
  // Thus, we need to `compose` all of the transforms so are each aplied
  std::vector<TransformFunction> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);

  // Create label loader
  auto targets = std::make_shared<Loader<uint64_t>>(labels, loadLabel);

  // Merge image and labels
  ds_ = std::make_shared<MergeDataset>(MergeDataset({transformed, targets}));
}

std::vector<af::array> ImageDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
    size_t free;
    size_t total;
    cudaMemGetInfo(&free, &total);
    auto limit = 1000 << 20;
    while(free <= limit) {
      std::cout<< "Device OOO, garbage collecting: " << std::this_thread::get_id()<<" :: "<< std::endl;
      af::deviceGC();
      std::this_thread::sleep_for(std::chrono::milliseconds(2000));
      cudaMemGetInfo(&free, &total);
    }
  return ds_->get(idx);
}

int64_t ImageDataset::size() const {
  return ds_->size();
}

Dataset::TransformFunction ImageDataset::resizeTransform(const uint64_t resize) {
  return [resize](const af::array& in) {
    return resizeSmallest(in, resize);
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
    const int cropLeft = (w - size) / 2;
    const int cropTop = (h - size) / 2;
    return in(
        af::seq(cropLeft, cropLeft + size - 1),
        af::seq(cropTop, cropTop + size - 1),
        af::span
    );
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

Dataset::TransformFunction ImageDataset::randomResizeTransform(
    const int low, const int high) {
  return [low, high](const af::array& in) {
    const float scale = float(rand()) / float(RAND_MAX);
    const int resize = low + (high - low) *  scale;
    return resizeSmallest(in, resize);
  };
};

Dataset::TransformFunction ImageDataset::randomCropTransform(
    const int tw,
    const int th) {
  return [th, tw](const af::array& in) {
    af::array out = in;
    const uint64_t w = in.dims(0);
    const uint64_t h = in.dims(1);
  const int x = rand() % (w - tw + 1);
	const int y = rand() % (h - th + 1);
	out = out(
  af::seq(x, x + tw - 1), af::seq(y, y + th - 1), af::span, af::span);
  return out;
  };
};

Dataset::TransformFunction ImageDataset::normalizeImage(
    const std::vector<float>& meanVector,
    const std::vector<float>& stdVector) {
  const af::array mean(1, 1, 3, 1, meanVector.data());
  const af::array std(1, 1, 3, 1, stdVector.data());
  return [mean, std](const af::array& in) {
    af::array out = in / 255.0f;
    out = af::batchFunc(out, mean, af::operator-);
    out = af::batchFunc(out, std, af::operator/);
    return out;
  };
};

} // namespace fl
