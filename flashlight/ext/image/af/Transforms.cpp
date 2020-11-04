/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/af/Transforms.h"

#include <numeric>
#include <random>

// TODO consider moving these outside of annonymous namespace
namespace {

float randomFloat(float a, float b) {
  float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return a + (b - a) * r;
}

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
  return af::resize(in, tw, th, AF_INTERP_BILINEAR);
}

af::array resize(const af::array& in, const int resize) {
  return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
}

af::array
crop(const af::array& in, const int x, const int y, const int w, const int h) {
  return in(af::seq(x, x + w - 1), af::seq(y, y + h - 1), af::span, af::span);
}

af::array centerCrop(const af::array& in, const int size) {
  const int w = in.dims(0);
  const int h = in.dims(1);
  const int cropLeft = std::round((static_cast<float>(w) - size) / 2.);
  const int cropTop = std::round((static_cast<float>(h) - size) / 2.);
  return crop(in, cropLeft, cropTop, size, size);
}

} // namespace

namespace fl {
namespace ext {
namespace image {

ImageTransform resizeTransform(const uint64_t resize) {
  return [resize](const af::array& in) { return resizeSmallest(in, resize); };
}

ImageTransform compose(std::vector<ImageTransform> transformfns) {
  return [transformfns](const af::array& in) {
    af::array out = in;
    for (auto fn : transformfns) {
      out = fn(out);
    }
    return out;
  };
}

ImageTransform centerCropTransform(const int size) {
  return [size](const af::array& in) { return centerCrop(in, size); };
};

ImageTransform randomHorizontalFlipTransform(const float p) {
  return [p](const af::array& in) {
    af::array out = in;
    if (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) > p) {
      const uint64_t w = in.dims(0);
      out = out(af::seq(w - 1, 0, -1), af::span, af::span, af::span);
    }
    return out;
  };
};

ImageTransform randomResizeCropTransform(
    const int size,
    const float scaleLow,
    const float scaleHigh,
    const float ratioLow,
    const float ratioHigh) {
  return [=](const af::array& in) mutable {
    const int w = in.dims(0);
    const int h = in.dims(1);
    const float area = w * h;
    for (int i = 0; i < 10; i++) {
      const float scale = randomFloat(scaleLow, scaleHigh);
      const float logRatio =
          randomFloat(std::log(ratioLow), std::log(ratioHigh));
      ;
      const float targetArea = scale * area;
      const float targetRatio = std::exp(logRatio);
      const int tw = std::round(std::sqrt(targetArea * targetRatio));
      const int th = std::round(std::sqrt(targetArea / targetRatio));
      if (0 < tw && tw <= w && 0 < th && th <= h) {
        const int x = std::rand() % (w - tw + 1);
        const int y = std::rand() % (h - th + 1);
        return resize(crop(in, x, y, tw, th), size);
      }
    }
    return centerCrop(resizeSmallest(in, size), size);
    ;
  };
}

ImageTransform randomResizeTransform(const int low, const int high) {
  return [low, high](const af::array& in) {
    const float scale =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    const int resize = low + (high - low) * scale;
    return resizeSmallest(in, resize);
  };
};

ImageTransform randomCropTransform(const int tw, const int th) {
  return [th, tw](const af::array& in) {
    af::array out = in;
    const uint64_t w = in.dims(0);
    const uint64_t h = in.dims(1);
    if (th > h || tw > w) {
      throw std::runtime_error(
          "Target th and target width are great the image size");
    }
    const int x = std::rand() % (w - tw + 1);
    const int y = std::rand() % (h - th + 1);
    return crop(in, x, y, tw, th);
  };
};

ImageTransform normalizeImage(
    const std::vector<float>& meanVector,
    const std::vector<float>& stdVector) {
  const af::array mean(1, 1, 3, 1, meanVector.data());
  const af::array std(1, 1, 3, 1, stdVector.data());
  return [mean, std](const af::array& in) {
    af::array out = in.as(f32) / 255.f;
    out = af::batchFunc(out, mean, af::operator-);
    out = af::batchFunc(out, std, af::operator/);
    return out;
  };
};

} // namespace image
} // namespace ext
} // namespace fl
