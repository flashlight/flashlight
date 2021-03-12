/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/ext/image/af/Transforms.h"

#include <ctime>
#include <numeric>
#include <random>

#include <iostream>

// TODO consider moving these outside of annonymous namespace
namespace {

float randomFloat(float a, float b) {
  float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return a + (b - a) * r;
}

template <class T>
T randomNegate(T a) {
  float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return r > 0.5 ? a : -a;
}

const float pi = std::acos(-1);

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
      // const uint64_t w = in.dims(1);
      // out = out(af::span, af::seq(w - 1, 0, -1), af::span, af::span);
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

ImageTransform randomEraseTransform(
    const float p,
    const float areaRatioMin,
    const float areaRatioMax,
    const float edgeRatioMin,
    const float edgeRatioMax) {
  return [p, areaRatioMin, areaRatioMax, edgeRatioMin, edgeRatioMax](
             const af::array& in) {
    if (p < static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) {
      // std::cout << 1 << std::endl;
      return in;
    }

    const int w = in.dims(0);
    const int h = in.dims(1);
    const int c = in.dims(2);

    af::array out = in;
    for (int i = 0; i < 10; i++) {
      float s = w * h * randomFloat(areaRatioMin, areaRatioMax);
      float r =
          std::exp(randomFloat(std::log(edgeRatioMin), std::log(edgeRatioMax)));
      int maskW = std::round(std::sqrt(s * r));
      int maskH = std::round(std::sqrt(s / r));
      if (maskW >= w || maskH >= h) {
        continue;
      }

      const int x = std::rand() % (w - maskW);
      const int y = std::rand() % (h - maskH);
      // std::cout << x << " " << y << " " << maskW << " " << maskH <<
      // std::endl; af::array fillValue =
      //     (af::randu(af::dim4(maskW, maskH, c)) * 255).as(in.type());
      af::array fillValue = af::randn(af::dim4(maskW, maskH, c), in.type());

      out(af::seq(x, x + maskW - 1),
          af::seq(y, y + maskH - 1),
          af::span,
          af::span) = fillValue;
      break;
    }
    out.eval();
    // std::cout << af::allTrue<bool>(out == in) << std::endl;
    return out;
  };
};

ImageTransform randomAugmentationTransform(const float p, const int n) {
  const std::vector<float> mean = {0.485 * 256, 0.456 * 256, 0.406 * 256};
  auto meanArr = af::array(1, 1, 3, 1, mean.data());
  float delta = 1e-2;

  return [p, n, delta, meanArr](const af::array& in) {
    // std::srand(std::time(nullptr));
    const int w = in.dims(0);
    const int h = in.dims(1);
    const int c = in.dims(2);
    auto background = af::tile(meanArr, w, h);

    auto res = in.as(f32);
    for (int i = 0; i < n; i++) {
      if (p < static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) {
        continue;
      }

      int mode = std::floor(randomFloat(0, 15 - 1e-5));
      // std::cout << mode << std::endl;
      if (mode < 1) {
        // rotate
        float theta = pi / 6 + randomFloat(-0.1, 0.1);
        theta = randomNegate(theta);
        res = af::rotate(res + delta, theta);

        auto mask = af::sum(res, 2) == 0;
        mask = af::tile(mask, 1, 1, 3).as(f32);
        res = mask * background + (1 - mask) * res - delta;
      } else if (mode < 2) {
        // skew
        float theta = pi / 6 + randomFloat(-0.1, 0.1);
        theta = randomNegate(theta);
        res = af::skew(res + delta, theta, 0);

        auto mask = af::sum(res, 2) == 0;
        mask = af::tile(mask, 1, 1, 3).as(f32);
        res = mask * background + (1 - mask) * res - delta;
      } else if (mode < 3) {
        // skew
        float theta = pi / 6 + randomFloat(-0.1, 0.1);
        theta = randomNegate(theta);
        res = af::skew(res + delta, 0, theta);

        auto mask = af::sum(res, 2) == 0;
        mask = af::tile(mask, 1, 1, 3).as(f32);
        res = mask * background + (1 - mask) * res - delta;
      } else if (mode < 4) {
        // translate
        int shift = 100 + randomFloat(-10, 10);
        shift = randomNegate(shift);
        res = af::translate(res + delta, shift, 0);

        auto mask = af::sum(res, 2) == 0;
        mask = af::tile(mask, 1, 1, 3).as(f32);
        res = mask * background + (1 - mask) * res - delta;
      } else if (mode < 5) {
        // translate
        int shift = 100 + randomFloat(-10, 10);
        shift = randomNegate(shift);
        res = af::translate(res + delta, 0, shift);

        auto mask = af::sum(res, 2) == 0;
        mask = af::tile(mask, 1, 1, 3).as(f32);
        res = mask * background + (1 - mask) * res - delta;
      } else if (mode < 6) {
        // color
        float enhance = .8 + randomFloat(-0.1, 0.1);
        enhance = 1 + randomNegate(enhance);
        auto meanPic = af::mean(res, 2);
        meanPic = af::tile(meanPic, af::dim4(1, 1, c));
        res = meanPic + enhance * (res - meanPic);
      } else if (mode < 7) {
        // auto contrast
        auto minPic = res;
        auto maxPic = res;
        for (int j = 0; j < 3; j++) {
          minPic = af::min(minPic, j);
          maxPic = af::max(maxPic, j);
        }
        minPic = af::tile(minPic, af::dim4(w, h, c));
        maxPic = af::tile(maxPic, af::dim4(w, h, c));
        auto scale = 256. / (maxPic - minPic + 1);
        res = scale * (res - minPic);
      } else if (mode < 8) {
        // contrast
        float enhance = .8 + randomFloat(-0.1, 0.1);
        enhance = 1 + randomNegate(enhance);
        auto meanPic = res;
        for (int j = 0; j < 3; j++) {
          meanPic = af::mean(meanPic, j);
        }
        meanPic = af::tile(meanPic, af::dim4(w, h, c));
        res = meanPic + enhance * (res - meanPic);
      } else if (mode < 9) {
        // brightness
        float enhance = .8 + randomFloat(-0.1, 0.1);
        enhance = 1 + randomNegate(enhance);
        res = res * enhance;
      } else if (mode < 10) {
        // invert
        res = 255 - res;
      } else if (mode < 11) {
        // solarize
        auto mask = (res < 26.).as(f32);
        res = mask * res + (1 - mask) * (255 - res);
      } else if (mode < 12) {
        // solarize
        auto mask = (res < 128.).as(f32);
        res = mask * (res + 100) + (1 - mask) * res;
      } else if (mode < 13) {
        // equalize
        res = flat(res);
        auto hist = af::histogram(res, 256);
        res = af::histEqual(res, hist);
        res = moddims(res, in.dims());
      } else if (mode < 14) {
        // posterize
        int mask = ~127; // 0x1f
        res = res.as(s32) & mask;
      } else if (mode < 15) {
        // sharpness
        float enhance = 1.7 + randomFloat(-0.2, 0.2);
        enhance = randomNegate(enhance);

        auto meanPic = af::mean(res, 2);
        auto blurKernel = af::gaussianKernel(7, 7);
        auto blur = af::convolve(meanPic, blurKernel);
        auto diff = af::tile(meanPic - blur, af::dim4(1, 1, c));

        res = res + enhance * diff;
      }

      res = res.as(f32);
      res = af::clamp(res, 0., 255.);
    }
    // std::cout << af::allTrue<bool>(res == in) << std::endl;
    // res = res.as(in.type());
    return res;
  };
}

} // namespace image
} // namespace ext
} // namespace fl
