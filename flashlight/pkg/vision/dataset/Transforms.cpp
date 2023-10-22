/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/Transforms.h"

#include <numeric>
#include <random>
#include <stdexcept>

#include "flashlight/fl/autograd/tensor/AutogradOps.h"
#include "flashlight/fl/nn/Utils.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Random.h"
#include "flashlight/pkg/vision/tensor/VisionOps.h"

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

namespace fl::pkg::vision {

Tensor resizeSmallest(const Tensor& in, const int resize) {
  const int w = in.dim(0);
  const int h = in.dim(1);
  int th, tw;
  if (h > w) {
    th = (resize * h) / w;
    tw = resize;
  } else {
    th = resize;
    tw = (resize * w) / h;
  }
  return fl::resize(in, {tw, th}, InterpolationMode::Bilinear);
}

Tensor resize(const Tensor& in, const int resize) {
  return fl::resize(in, {resize, resize}, InterpolationMode::Bilinear);
}

Tensor
crop(const Tensor& in, const int x, const int y, const int w, const int h) {
  return in(fl::range(x, x + w), fl::range(y, y + h));
}

Tensor centerCrop(const Tensor& in, const int size) {
  const int w = in.dim(0);
  const int h = in.dim(1);
  const int cropLeft = std::round((static_cast<float>(w) - size) / 2.);
  const int cropTop = std::round((static_cast<float>(h) - size) / 2.);
  return crop(in, cropLeft, cropTop, size, size);
}

Tensor rotate(const Tensor& input, const float theta, const Tensor& fillImg) {
  return fl::rotate(input, theta, fillImg);
}

Tensor skewX(const Tensor& input, const float theta, const Tensor& fillImg) {
  return fl::shear(input, {theta, 0}, {}, fillImg);
}

Tensor skewY(const Tensor& input, const float theta, const Tensor& fillImg) {
  return fl::shear(input, {0, theta}, {}, fillImg);
}

Tensor translateX(const Tensor& input, const int shift, const Tensor& fillImg) {
  return fl::translate(input, {shift, 0}, {}, fillImg);
}

Tensor translateY(const Tensor& input, const int shift, const Tensor& fillImg) {
  return fl::translate(input, {0, shift}, {}, fillImg);
}

Tensor colorEnhance(const Tensor& input, const float enhance) {
  auto c = input.dim(2);
  auto meanPic = fl::mean(input, {2}, /* keepDims = */ true);
  meanPic = fl::tile(meanPic, {1, 1, c});
  return meanPic + enhance * (input - meanPic);
}

Tensor autoContrast(const Tensor& input) {
  auto minPic = fl::amin(input);
  auto maxPic = fl::amax(input);
  if (fl::all(minPic == maxPic).asScalar<bool>()) {
    return input;
  }

  auto scale = fl::tile(255. / (maxPic - minPic), input.shape());
  minPic = fl::tile(minPic, input.shape());
  return scale * (input - minPic);
}

Tensor contrastEnhance(const Tensor& input, const float enhance) {
  auto meanPic = fl::mean(input);
  meanPic = fl::tile(meanPic, input.shape());
  return meanPic + enhance * (input - meanPic);
}

Tensor brightnessEnhance(const Tensor& input, const float enhance) {
  return input * enhance;
}

Tensor invert(const Tensor& input) {
  return 255. - input;
}

Tensor solarize(const Tensor& input, const float threshold) {
  auto mask = (input < threshold);
  return mask * input + (1 - mask) * (255 - input);
}

Tensor
solarizeAdd(const Tensor& input, const float threshold, const float addValue) {
  auto mask = (input < threshold);
  return mask * (input + addValue) + (1 - mask) * input;
}

Tensor equalize(const Tensor& input) {
  auto res = input;
  for (int i = 0; i < 3; i++) {
    auto resSlice = res(fl::span, fl::span, i);
    auto hist = fl::histogram(
        resSlice, /* numBins = */ 256, /* minVal = */ 0, /* maxVal = */ 255);
    res(fl::span, fl::span, i) = fl::equalize(resSlice, hist);
  }
  return res;
}

Tensor posterize(const Tensor& input, const int bitsToKeep) {
  if (bitsToKeep < 1 || bitsToKeep > 8) {
    throw std::invalid_argument("bitsToKeep needs to be in [1, 8]");
  }
  uint8_t mask = ~((1 << (8 - bitsToKeep)) - 1);
  auto res = input.astype(fl::dtype::u8) && mask;
  return res.astype(input.type());
}

Tensor sharpnessEnhance(const Tensor& input, const float enhance) {
  const int w = input.dim(0);
  const int h = input.dim(1);
  const int c = input.dim(2);
  const int kernelSize = 7;
  const int stride = 1;
  const int samePad = static_cast<int>(PaddingMode::SAME);

  auto meanPic = fl::mean(input, {2});
  auto blurKernel = fl::gaussianFilter({kernelSize, kernelSize});
  auto blur = fl::conv2d(
      fl::reshape(meanPic, {w, h, 1, 1}),
      blurKernel,
      /* sx = */ stride,
      /* sy = */ stride,
      // ensure output size is the same as input size
      /* px = */ derivePadding(w, kernelSize, 1, samePad, 1),
      /* py = */ derivePadding(h, kernelSize, 1, samePad, 1));
  blur = fl::reshape(blur, {w, h});
  auto diff = fl::tile(meanPic - blur, {1, 1, c});
  return input + enhance * diff;
}

Tensor oneHot(
    const Tensor& targets,
    const int numClasses,
    const float labelSmoothing) {
  float offValue = labelSmoothing / numClasses;
  float onValue = 1. - labelSmoothing;

  int X = targets.elements();
  auto y = fl::reshape(targets, {1, X});
  auto A = fl::arange({numClasses, X});
  auto B = fl::tile(y, {numClasses});
  auto mask = A == B; // [C X]

  Tensor out = fl::full({numClasses, X}, onValue);
  out = out * mask + offValue;

  return out;
}

std::pair<Tensor, Tensor> mixupBatch(
    const float lambda,
    const Tensor& input,
    const Tensor& target,
    const int numClasses,
    const float labelSmoothing) {
  // in : W x H x C x B
  // target: B x 1
  auto targetOneHot = oneHot(target, numClasses, labelSmoothing);
  if (lambda == 0) {
    return {input, targetOneHot};
  }

  // mix input
  auto inputFlipped = fl::flip(input, 3);
  auto inputMixed = lambda * inputFlipped + (1 - lambda) * input;

  // mix target
  auto targetOneHotFlipped =
      oneHot(fl::flip(target, 0), numClasses, labelSmoothing);
  auto targetOneHotMixed =
      lambda * targetOneHotFlipped + (1 - lambda) * targetOneHot;

  return {inputMixed, targetOneHotMixed};
}

std::pair<Tensor, Tensor> cutmixBatch(
    const float lambda,
    const Tensor& input,
    const Tensor& target,
    const int numClasses,
    const float labelSmoothing) {
  // in : W x H x C x B
  // target: B x 1
  auto targetOneHot = oneHot(target, numClasses, labelSmoothing);
  if (lambda == 0) {
    return {input, targetOneHot};
  }

  // mix input
  auto inputFlipped = fl::flip(input, 3);

  const float lambdaSqrt = std::sqrt(lambda);
  const int w = input.dim(0);
  const int h = input.dim(1);
  const int maskW = std::round(w * lambdaSqrt);
  const int maskH = std::round(h * lambdaSqrt);
  const int centerW = randomFloat(0, w);
  const int centerH = randomFloat(0, h);

  const int x1 = std::max(0, centerW - maskW / 2);
  const int x2 = std::min(w, centerW + maskW / 2 + 1);
  const int y1 = std::max(0, centerH - maskH / 2);
  const int y2 = std::min(h, centerH + maskH / 2 + 1);

  auto inputMixed = input;
  inputMixed(fl::range(x1, x2), fl::range(y1, y2)) =
      inputFlipped(fl::range(x1, x2), fl::range(y1, y2));
  auto newLambda = static_cast<float>(x2 - x1) * (y2 - y1) / (w * h);

  // mix target
  auto targetOneHotFlipped =
      oneHot(fl::flip(target, 0), numClasses, labelSmoothing);
  auto targetOneHotMixed =
      newLambda * targetOneHotFlipped + (1 - newLambda) * targetOneHot;

  return {inputMixed, targetOneHotMixed};
}

ImageTransform resizeTransform(const uint64_t resize) {
  return [resize](const Tensor& in) { return resizeSmallest(in, resize); };
}

ImageTransform compose(std::vector<ImageTransform> transformfns) {
  return [transformfns](const Tensor& in) {
    Tensor out = in;
    for (const auto& fn : transformfns) {
      out = fn(out);
    }
    return out;
  };
}

ImageTransform centerCropTransform(const int size) {
  return [size](const Tensor& in) { return centerCrop(in, size); };
};

ImageTransform randomHorizontalFlipTransform(const float p) {
  return [p](const Tensor& in) {
    Tensor out = in;
    if (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) > p) {
      const long long w = in.dim(0);
      // reverse indices - w --> 0 - TODO: use fl::flip?
      out = out(fl::range(w - 1, 1, -1));
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
  return [=](const Tensor& in) mutable {
    const int w = in.dim(0);
    const int h = in.dim(1);
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
  };
}

ImageTransform randomResizeTransform(const int low, const int high) {
  return [low, high](const Tensor& in) {
    const float scale =
        static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    const int resize = low + (high - low) * scale;
    return resizeSmallest(in, resize);
  };
};

ImageTransform randomCropTransform(const int tw, const int th) {
  return [th, tw](const Tensor& in) {
    Tensor out = in;
    const uint64_t w = in.dim(0);
    const uint64_t h = in.dim(1);
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
  const Tensor mean = Tensor::fromVector({1, 1, 3}, meanVector);
  const Tensor std = Tensor::fromVector({1, 1, 3}, stdVector);
  return [mean, std](const Tensor& in) {
    Tensor out = in.astype(fl::dtype::f32) / 255.f;
    out = out - mean;
    out = out / std;
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
             const Tensor& in) {
    if (p < randomFloat(0, 1)) {
      return in;
    }

    const float epsilon = 1e-7;
    const int w = in.dim(0);
    const int h = in.dim(1);
    const int c = in.dim(2);

    Tensor out = in;
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
      Tensor fillValue = fl::randn({maskW, maskH, c}, in.type());

      out(fl::range(x, x + maskW), fl::range(y, y + maskH)) = fillValue;
      break;
    }
    return out;
  };
};

ImageTransform randomAugmentationDeitTransform(
    const float p,
    const int n,
    const Tensor& fillImg) {
  // Selected 15 transform functions with specific parameters
  // following https://git.io/JYGG6

  return [p, n, fillImg](const Tensor& in) {
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

        res = fl::rotate(res, theta, fillImg);
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
      res = fl::clip(res, 0., 255.).astype(res.type());
    }
    return res;
  };
}

} // namespace fl
