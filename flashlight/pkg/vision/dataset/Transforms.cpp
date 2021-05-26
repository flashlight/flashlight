/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/Transforms.h"

#include <numeric>
#include <random>
#include <stdexcept>

// TODO consider moving these outside of annonymous namespace
namespace {

float randomFloat(float a, float b) {
  float r = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  return a + (b - a) * r;
}

template <class T>
T randomNegate(T a) {
  float r = randomFloat(0, 1);
  return r > 0.5 ? a : -a;
}

template <class T>
T randomPerturbNegate(T base, T minNoise, T maxNoise) {
  float noise = randomFloat(minNoise, maxNoise);
  return randomNegate(base + noise);
}

} // namespace

namespace fl {
namespace pkg {
namespace vision {

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

// TODO: for af::rotate, af::skew, af::translate
//  - they only support zero-filling on empty spots
//  - we add a hack to manually fill from fillImg
//  - to be removed once customized filling is supported in AF directly
af::array
rotate(const af::array& input, const float theta, const af::array& fillImg) {
  float delta = 1e-2;
  auto res = input;
  if (!fillImg.isempty()) {
    res = res + delta;
  }

  res = af::rotate(res, theta);

  if (!fillImg.isempty()) {
    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3);
    res = mask * fillImg + (1 - mask) * (res - delta);
  }
  return res;
}

af::array
skewX(const af::array& input, const float theta, const af::array& fillImg) {
  float delta = 1e-2;
  auto res = input;
  if (!fillImg.isempty()) {
    res = res + delta;
  }

  res = af::skew(res, theta, 0);

  if (!fillImg.isempty()) {
    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3);
    res = mask * fillImg + (1 - mask) * (res - delta);
  }
  return res;
}

af::array
skewY(const af::array& input, const float theta, const af::array& fillImg) {
  float delta = 1e-2;
  auto res = input;
  if (!fillImg.isempty()) {
    res = res + delta;
  }

  res = af::skew(res, 0, theta);

  if (!fillImg.isempty()) {
    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3);
    res = mask * fillImg + (1 - mask) * (res - delta);
  }
  return res;
}

af::array
translateX(const af::array& input, const int shift, const af::array& fillImg) {
  float delta = 1e-2;
  auto res = input;
  if (!fillImg.isempty()) {
    res = res + delta;
  }

  res = af::translate(res, shift, 0);

  if (!fillImg.isempty()) {
    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3);
    res = mask * fillImg + (1 - mask) * (res - delta);
  }
  return res;
}

af::array
translateY(const af::array& input, const int shift, const af::array& fillImg) {
  float delta = 1e-2;
  auto res = input;
  if (!fillImg.isempty()) {
    res = res + delta;
  }

  res = af::translate(res, 0, shift);

  if (!fillImg.isempty()) {
    auto mask = af::sum(res, 2) == 0;
    mask = af::tile(mask, 1, 1, 3);
    res = mask * fillImg + (1 - mask) * (res - delta);
  }
  return res;
}

af::array colorEnhance(const af::array& input, const float enhance) {
  auto c = input.dims(2);
  auto meanPic = af::mean(input, 2);
  meanPic = af::tile(meanPic, af::dim4(1, 1, c));
  return meanPic + enhance * (input - meanPic);
}

af::array autoContrast(const af::array& input) {
  auto inputFlat = af::flat(input);
  auto minPic = af::min(inputFlat, 0);
  auto maxPic = af::max(inputFlat, 0);
  if (af::allTrue<bool>(minPic == maxPic)) {
    return input;
  }

  auto scale = af::tile(255. / (maxPic - minPic), input.dims());
  minPic = af::tile(minPic, input.dims());
  return scale * (input - minPic);
}

af::array contrastEnhance(const af::array& input, const float enhance) {
  auto inputFlat = af::flat(input);
  auto meanPic = af::mean(inputFlat, 0);
  meanPic = af::tile(meanPic, input.dims());
  return meanPic + enhance * (input - meanPic);
}

af::array brightnessEnhance(const af::array& input, const float enhance) {
  return input * enhance;
}

af::array invert(const af::array& input) {
  return 255. - input;
}

af::array solarize(const af::array& input, const float threshold) {
  auto mask = (input < threshold);
  return mask * input + (1 - mask) * (255 - input);
}

af::array solarizeAdd(
    const af::array& input,
    const float threshold,
    const float addValue) {
  auto mask = (input < threshold);
  return mask * (input + addValue) + (1 - mask) * input;
}

af::array equalize(const af::array& input) {
  auto res = input;
  for (int i = 0; i < 3; i++) {
    auto resSlice = res(af::span, af::span, i);
    auto hist = af::histogram(resSlice, 256, 0, 255);
    resSlice = af::histEqual(resSlice, hist);
  }
  return res;
}

af::array posterize(const af::array& input, const int bitsToKeep) {
  if (bitsToKeep < 1 || bitsToKeep > 8) {
    throw std::invalid_argument("bitsToKeep needs to be in [1, 8]");
  }
  uint8_t mask = ~((1 << (8 - bitsToKeep)) - 1);
  auto res = input.as(u8) & mask;
  return res.as(input.type());
}

af::array sharpnessEnhance(const af::array& input, const float enhance) {
  const int c = input.dims(2);

  auto meanPic = af::mean(input, 2);
  auto blurKernel = af::gaussianKernel(7, 7);
  auto blur = af::convolve(meanPic, blurKernel);
  auto diff = af::tile(meanPic - blur, af::dim4(1, 1, c));
  return input + enhance * diff;
}

af::array oneHot(
    const af::array& targets,
    const int numClasses,
    const float labelSmoothing) {
  float offValue = labelSmoothing / numClasses;
  float onValue = 1. - labelSmoothing;

  int X = targets.elements();
  auto y = af::moddims(targets, af::dim4(1, X));
  auto A = af::range(af::dim4(numClasses, X));
  auto B = af::tile(y, af::dim4(numClasses));
  auto mask = A == B; // [C X]

  af::array out = af::constant(onValue, af::dim4(numClasses, X));
  out = out * mask + offValue;

  return out;
}

std::pair<af::array, af::array> mixupBatch(
    const float lambda,
    const af::array& input,
    const af::array& target,
    const int numClasses,
    const float labelSmoothing) {
  // in : W x H x C x B
  // target: B x 1
  auto targetOneHot = oneHot(target, numClasses, labelSmoothing);
  if (lambda == 0) {
    return {input, targetOneHot};
  }

  // mix input
  auto inputFlipped = af::flip(input, 3);
  auto inputMixed = lambda * inputFlipped + (1 - lambda) * input;

  // mix target
  auto targetOneHotFlipped =
      oneHot(af::flip(target, 0), numClasses, labelSmoothing);
  auto targetOneHotMixed =
      lambda * targetOneHotFlipped + (1 - lambda) * targetOneHot;

  return {inputMixed, targetOneHotMixed};
}

std::pair<af::array, af::array> cutmixBatch(
    const float lambda,
    const af::array& input,
    const af::array& target,
    const int numClasses,
    const float labelSmoothing) {
  // in : W x H x C x B
  // target: B x 1
  auto targetOneHot = oneHot(target, numClasses, labelSmoothing);
  if (lambda == 0) {
    return {input, targetOneHot};
  }

  // mix input
  auto inputFlipped = af::flip(input, 3);

  const float lambdaSqrt = std::sqrt(lambda);
  const int w = input.dims(0);
  const int h = input.dims(1);
  const int maskW = std::round(w * lambdaSqrt);
  const int maskH = std::round(h * lambdaSqrt);
  const int centerW = randomFloat(0, w);
  const int centerH = randomFloat(0, h);

  const int x1 = std::max(0, centerW - maskW / 2);
  const int x2 = std::min(w - 1, centerW + maskW / 2);
  const int y1 = std::max(0, centerH - maskH / 2);
  const int y2 = std::min(h - 1, centerH + maskH / 2);

  auto inputMixed = input;
  inputMixed(af::seq(x1, x2), af::seq(y1, y2), af::span, af::span) =
      inputFlipped(af::seq(x1, x2), af::seq(y1, y2), af::span, af::span);
  auto newLambda = static_cast<float>(x2 - x1 + 1) * (y2 - y1 + 1) / (w * h);

  // mix target
  auto targetOneHotFlipped =
      oneHot(af::flip(target, 0), numClasses, labelSmoothing);
  auto targetOneHotMixed =
      newLambda * targetOneHotFlipped + (1 - newLambda) * targetOneHot;

  return {inputMixed, targetOneHotMixed};
}

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

ImageTransform randomEraseTransform(
    const float p,
    const float areaRatioMin,
    const float areaRatioMax,
    const float edgeRatioMin,
    const float edgeRatioMax) {
  // follows: https://git.io/JY9R7
  return [p, areaRatioMin, areaRatioMax, edgeRatioMin, edgeRatioMax](
             const af::array& in) {
    if (p < randomFloat(0, 1)) {
      return in;
    }

    const float epsilon = 1e-7;
    const int w = in.dims(0);
    const int h = in.dims(1);
    const int c = in.dims(2);

    af::array out = in;
    for (int i = 0; i < 10; i++) {
      const float s = w * h * randomFloat(areaRatioMin, areaRatioMax);
      const float r =
          std::exp(randomFloat(std::log(edgeRatioMin), std::log(edgeRatioMax)));
      const int maskW = std::round(std::sqrt(s * r));
      const int maskH = std::round(std::sqrt(s / r));
      if (maskW >= w || maskH >= h) {
        continue;
      }

      const int x = static_cast<int>(randomFloat(0, w - maskW - epsilon));
      const int y = static_cast<int>(randomFloat(0, h - maskH - epsilon));
      af::array fillValue = af::randn(af::dim4(maskW, maskH, c), in.type());

      out(af::seq(x, x + maskW - 1),
          af::seq(y, y + maskH - 1),
          af::span,
          af::span) = fillValue;
      break;
    }
    return out;
  };
};

ImageTransform randomAugmentationDeitTransform(
    const float p,
    const int n,
    const af::array& fillImg) {
  // Selected 15 transform functions with specific parameters
  // following https://git.io/JYGG6

  return [p, n, fillImg](const af::array& in) {
    auto res = in;
    for (int i = 0; i < n; i++) {
      if (p < randomFloat(0, 1)) {
        continue;
      }

      int mode = std::floor(randomFloat(0, 15 - 1e-5));
      if (mode == 0) {
        // rotate
        float baseTheta = .47;
        float theta = randomPerturbNegate<float>(baseTheta, -0.02, 0.02);

        res = rotate(res, theta, fillImg);
      } else if (mode == 1) {
        // skew-x
        float baseTheta = .27;
        float theta = randomPerturbNegate<float>(baseTheta, -0.02, 0.02);

        res = skewX(res, theta, fillImg);
      } else if (mode == 2) {
        // skew-y
        float baseTheta = .27;
        float theta = randomPerturbNegate<float>(baseTheta, -0.02, 0.02);

        res = skewY(res, theta, fillImg);
      } else if (mode == 3) {
        // translate-x
        int baseDelta = 90;
        int delta = randomPerturbNegate<int>(baseDelta, -3, 3);

        res = translateX(res, delta, fillImg);
      } else if (mode == 4) {
        // translate-y
        int baseDelta = 90;
        int delta = randomPerturbNegate<int>(baseDelta, -3, 3);

        res = translateY(res, delta, fillImg);
      } else if (mode == 5) {
        // color
        float baseEnhance = .8;
        float enhance =
            1 + randomPerturbNegate<float>(baseEnhance, -0.03, 0.03);

        res = colorEnhance(res, enhance);
      } else if (mode == 6) {
        // auto contrast
        res = autoContrast(res);
      } else if (mode == 7) {
        // contrast
        float baseEnhance = .8;
        float enhance =
            1 + randomPerturbNegate<float>(baseEnhance, -0.03, 0.03);

        res = contrastEnhance(res, enhance);
      } else if (mode == 8) {
        // brightness
        float baseEnhance = .8;
        float enhance =
            1 + randomPerturbNegate<float>(baseEnhance, -0.03, 0.03);

        res = brightnessEnhance(res, enhance);
      } else if (mode == 9) {
        // invert
        res = invert(res);
      } else if (mode == 10) {
        // solarize
        res = solarize(res, 26.);
      } else if (mode == 11) {
        // solarize add
        res = solarizeAdd(res, 128., 100.);
      } else if (mode == 12) {
        // equalize
        res = equalize(res);
      } else if (mode == 13) {
        // posterize
        res = posterize(res, 1);
      } else if (mode == 14) {
        // sharpness
        float baseEnhance = .5;
        float enhance = randomPerturbNegate<float>(baseEnhance, -0.01, 0.01);

        res = sharpnessEnhance(res, enhance);
      }
      res = af::clamp(res, 0., 255.).as(res.type());
    }
    return res;
  };
}

} // namespace vision
} // namespace pkg
} // namespace fl
