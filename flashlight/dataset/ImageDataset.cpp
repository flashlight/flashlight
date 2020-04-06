#include <thread>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#include <numeric>

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

class RandomFloatBetween {

public:
  RandomFloatBetween() = default;
  RandomFloatBetween(float low, float high) :
    random_engine_{std::make_shared<std::mt19937>(std::random_device{}())},
    distribution_(low, high) {}

  float operator()() {
    return distribution_(*random_engine_.get());
  }
private:
  std::shared_ptr<std::mt19937> random_engine_;
  std::uniform_real_distribution<float> distribution_;
};

/*
 * Resizes the smallest length edge of an image to be resize while keeping
 * the aspect ratio
 */
af::array resizeSmallest(const af::array& in, const int resize) {
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
    //return af::resize(in, tw, th, AF_INTERP_BICUBIC);
    return af::resize(in, tw, th, AF_INTERP_BILINEAR);
}

af::array resize(const af::array& in, const int resize) {
  return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
}

af::array crop(
    const af::array& in,
    const int x,
    const int y,
    const int w,
    const int h) {
	return in(af::seq(x, x + w - 1), af::seq(y, y + h - 1), af::span, af::span);
}

af::array centerCrop2(const af::array& in, const int size) {
    const int w = in.dims(0);
    const int h = in.dims(1);
    const int cropLeft = round((float(w) - size) / 2.);
    const int cropTop = round((float(h) - size) / 2.);
    return crop(in, cropLeft, cropTop, size, size);
}

/*
 * Loads a jpeg from filepath fp. Note: It will automatically convert from any
 * numnber of channels to create an array with 3 channels
 */
af::array loadJpeg(const std::string& fp) {
#if 0
  return af::loadImage(fp.c_str(), true);
  //af::array img;
  //try {
    ////return af::loadImage(fp.c_str(), true);
    //img = af::loadImageNative(fp.c_str());
  //} catch (...){
    //img = af::constant(0, 224, 244, 3);
    //std::cout << "Filepath " << fp << std::endl;
  //}
  //if(img.type() != u8) {
    //std::cout << img.type() << std::endl;
  //}
  //if (img.dims(2) == 3) {
    //return img;
  //} else if (img.dims(2) == 1) {
    //img = af::tile(img, 2, 3);
    //auto img2 = af::colorSpace(img, AF_RGB, AF_GRAY);
    //return img2;
  //}
#else
	int w, h, c;
  // STB image will automatically return desired_no_channels.
  // NB: c will be the original number of channels
  //std::cout << " filepath " << fp << std::endl;
	int desired_no_channels = 3;
	unsigned char *img = stbi_load(fp.c_str(), &w, &h, &c, desired_no_channels);
	if (img) {
		af::array result = af::array(desired_no_channels, w, h, img);
		stbi_image_free(img);
    return af::reorder(result, 1, 2, 0);
    //return result;
	} else {
    throw std::invalid_argument("Could not load from filepath" + fp);
	}
#endif
}

af::array loadLabel(const uint64_t x) {
  return af::constant(x, 1, 1, 1, 1, u64);
}

af::array loadNpArray(const uint64_t x, const std::string pytorchDump) {
  //std::string pytorchDump = "/private/home/padentomasello/tmp/pytorch_dump/save2/image";
  std::stringstream ss;
  ss << pytorchDump << "image" << x << ".bin";
  const std::string fp = ss.str();
  std::ifstream infile(fp, std::ios::binary);
  if(!infile) {
      throw std::invalid_argument("Could not read from fp" + fp);
  }
  std::vector<float> vec(224 * 224 *3);
  infile.read((char*) vec.data(), vec.size() * sizeof(float));
  if(!infile) {
    throw std::invalid_argument("Could not read from fp" + fp);
  }
  infile.close();
  return af::array(af::dim4(224, 224, 3), vec.data());
}

af::array loadNpLabel(const uint64_t x, const std::string pytorchDump) {
  //std::string pytorchDump = "/private/home/padentomasello/tmp/pytorch_dump/save2/label";
  std::stringstream ss;
  ss << pytorchDump << "label" << x << ".bin";
  const std::string fp = ss.str();
  std::ifstream infile(fp, std::ios::binary);
  if(!infile) {
      throw std::invalid_argument("Could not read from fp" + fp);
  }
  std::vector<int64_t> vec(1);
  infile.read((char*) vec.data(), vec.size() * sizeof(int64_t));
  if(!infile) {
    throw std::invalid_argument("Could not read from fp" + fp);
  }
  infile.close(); 
  return af::constant(vec[0], 1, 1, 1, 1, s64);
}

}

namespace fl {

NumpyDataset::NumpyDataset(int n, const std::string fp) : n_(n) {
  std::vector<int> indices(n_);
  std::iota(indices.begin(), indices.end(), 0);
  auto targets = std::make_shared<Loader<int>>(indices, [fp](const uint64_t x) {
      return loadNpLabel(x, fp); });
  auto images = std::make_shared<Loader<int>>(indices, [fp](const uint64_t x) {
      return loadNpArray(x, fp); });
  ds_ = std::make_shared<MergeDataset>(MergeDataset({images, targets}));
}

int64_t NumpyDataset::size() const {
  return n_;
}

std::vector<af::array> NumpyDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  return ds_->get(idx);
}

ImageDataset::ImageDataset(
    std::vector<std::string> filepaths,
    std::vector<uint64_t> labels,
    std::vector<TransformFunction>& transformfns
 ) {

  // Create image loader and apply transforms
  //std::vector<int> indices(90000);
  //std::iota(indices.begin(), indices.end(), 0);
  //auto targets = std::make_shared<Loader<int>>(indices, loadNpLabel);
  //auto images = std::make_shared<Loader<int>>(indices, loadNpArray);
  //auto transformed = images;
  // TransformDataset will apply each transform in a vector to the respective af::array
  // Thus, we need to `compose` all of the transforms so are each aplied
  auto images = std::make_shared<Loader<std::string>>(filepaths, loadJpeg);
  std::vector<TransformFunction> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);

  auto targets = std::make_shared<Loader<uint64_t>>(labels, loadLabel);
  // Create label loader

  // Merge image and labels
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
    return centerCrop2(in, size);
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

Dataset::TransformFunction ImageDataset::randomResizeCropTransform(
    const int size,
    const float scaleLow,
    const float scaleHigh,
    const float ratioLow,
    const float ratioHigh) {
   auto scaleDist = RandomFloatBetween(scaleLow, scaleHigh);
   auto ratioDist = RandomFloatBetween(log(ratioLow), log(ratioHigh));
  return [ size, scaleDist, ratioDist] (const af::array& in) mutable {
    const int w = in.dims(0);
    const int h = in.dims(1);
    const float area = w * h;
    for(int i = 0; i < 10; i++) {
      const float scale = scaleDist();
      const float ratio = ratioDist();
      const float targetArea = scale * area;
      const float targetRatio = exp(ratio);
      const int tw = round(sqrt(targetArea * targetRatio));
      const int th = round(sqrt(targetArea / targetRatio));
      if (0 < tw && tw <= w && 0 < th && th <= h) {
        const int x = rand() % (w - tw + 1);
        const int y = rand() % (h - th + 1);
        return resize(crop(in, x, y, tw, th), size);
      }
    }
    return centerCrop2(resizeSmallest(in, size), size);;
  };
}

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
    return crop(in, x, y, tw, th);
  };
};

Dataset::TransformFunction ImageDataset::normalizeImage(
    const std::vector<float>& meanVector,
    const std::vector<float>& stdVector) {
  const af::array mean(1, 1, 3, 1, meanVector.data());
  const af::array std(1, 1, 3, 1, stdVector.data());
  return [mean, std](const af::array& in) {
    af::array out = in.as(f32);
    out = af::batchFunc(out, mean, af::operator-);
    out = af::batchFunc(out, std, af::operator/);
    //af_print(out(af::span, 0, 0))
    return out;
  };
};

} // namespace fl
