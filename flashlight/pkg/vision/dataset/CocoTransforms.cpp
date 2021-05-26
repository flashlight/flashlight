/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/CocoTransforms.h"

#include <assert.h>

#include "flashlight/pkg/vision/dataset/BoxUtils.h"
#include "flashlight/pkg/vision/dataset/TransformAllDataset.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"

namespace {

int randomInt(int min, int max) {
  return std::rand() % (max - min + 1) + min;
}
} // namespace

namespace fl {
namespace pkg {
namespace vision {

std::vector<af::array>
crop(const std::vector<af::array>& in, int x, int y, int tw, int th) {
  const af::array& image = in[ImageIdx];
  const af::array croppedImage = fl::ext::image::crop(image, x, y, tw, th);

  const af::array& boxes = in[BboxesIdx];

  const std::vector<int> translateVector = {x, y, x, y};
  const std::vector<int> maxSizeVector = {tw, th};
  af::array targetSize = af::array(2, maxSizeVector.data());

  const af::array translateArray = af::array(4, translateVector.data());
  const af::array maxSizeArray = af::array(2, maxSizeVector.data());

  af::array croppedBoxes = boxes;
  af::array labels = in[ClassesIdx];

  if (!croppedBoxes.isempty()) {
    croppedBoxes = af::batchFunc(croppedBoxes, translateArray, af::operator-);
    croppedBoxes = af::moddims(croppedBoxes, {2, 2, boxes.dims(1)});
    croppedBoxes = af::batchFunc(croppedBoxes, maxSizeArray, af::min);
    croppedBoxes = af::max(croppedBoxes, 0.0);
    af::array keep = allTrue(
        croppedBoxes(af::span, af::seq(1, 1)) >
        croppedBoxes(af::span, af::seq(0, 0)));
    croppedBoxes = af::moddims(croppedBoxes, {4, boxes.dims(1)});
    croppedBoxes = croppedBoxes(af::span, keep);
    labels = labels(af::span, keep);
  }
  return {croppedImage,
          targetSize,
          in[ImageIdIdx],
          in[OriginalSizeIdx],
          croppedBoxes,
          labels};
};

std::vector<af::array> hflip(const std::vector<af::array>& in) {
  af::array image = in[ImageIdx];
  const int w = image.dims(0);
  image = image(af::seq(w - 1, 0, -1), af::span, af::span, af::span);

  af::array bboxes = in[BboxesIdx];
  if (!bboxes.isempty()) {
    af::array bboxes_flip = af::array(bboxes.dims());
    bboxes_flip(0, af::span) = (bboxes(2, af::span) * -1) + w;
    bboxes_flip(1, af::span) = bboxes(1, af::span);
    bboxes_flip(2, af::span) = (bboxes(0, af::span) * -1) + w;
    bboxes_flip(3, af::span) = bboxes(3, af::span);
    bboxes = bboxes_flip;
  }
  return {image,
          in[TargetSizeIdx],
          in[ImageIdIdx],
          in[OriginalSizeIdx],
          bboxes,
          in[ClassesIdx]};
}

std::vector<af::array> normalize(const std::vector<af::array>& in) {
  auto boxes = in[BboxesIdx];

  if (!boxes.isempty()) {
    auto image = in[ImageIdx];
    auto w = float(image.dims(0));
    auto h = float(image.dims(1));

    boxes = xyxy2cxcywh(boxes);
    const std::vector<float> ratioVector = {w, h, w, h};
    af::array ratioArray = af::array(4, ratioVector.data());
    boxes = af::batchFunc(boxes, ratioArray, af::operator/);
  }
  return {in[ImageIdx],
          in[TargetSizeIdx],
          in[ImageIdIdx],
          in[OriginalSizeIdx],
          boxes,
          in[ClassesIdx]};
}

std::vector<af::array>
randomResize(std::vector<af::array> inputs, int size, int maxsize) {
  auto getSize = [](const af::array& in, int size, int maxSize = 0) {
    int w = in.dims(0);
    int h = in.dims(1);
    // long size;
    if (maxSize > 0) {
      float minOriginalSize = std::min(w, h);
      float maxOriginalSize = std::max(w, h);
      if (maxOriginalSize / minOriginalSize * size > maxSize) {
        size = round(maxSize * minOriginalSize / maxOriginalSize);
      }
    }

    if ((w <= h && w == size) || (h <= w && h == size)) {
      return std::make_pair(w, h);
    }
    int ow, oh;
    if (w < h) {
      ow = size;
      oh = size * h / w;
    } else {
      oh = size;
      ow = size * w / h;
    }
    return std::make_pair(ow, oh);
  };

  af::array image = inputs[ImageIdx];
  auto output_size = getSize(image, size, maxsize);
  const af::dim4 originalDims = image.dims();
  af::array resizedImage;
  resizedImage = af::resize(
      image, output_size.first, output_size.second, AF_INTERP_BILINEAR);
  const af::dim4 resizedDims = resizedImage.dims();

  af::array boxes = inputs[BboxesIdx];
  if (!boxes.isempty()) {
    const float ratioWidth = float(resizedDims[0]) / float(originalDims[0]);
    const float ratioHeight = float(resizedDims[1]) / float(originalDims[1]);

    const std::vector<float> resizeVector = {
        ratioWidth, ratioHeight, ratioWidth, ratioHeight};
    af::array resizedArray = af::array(4, resizeVector.data());
    boxes = af::batchFunc(boxes, resizedArray, af::operator*);
  }

  long long int imageSizeArray[] = {resizedImage.dims(1), resizedImage.dims(0)};
  af::array sizeArray = af::array(2, imageSizeArray);
  return {resizedImage,
          sizeArray,
          inputs[ImageIdIdx],
          inputs[OriginalSizeIdx],
          boxes,
          inputs[ClassesIdx]};
}

TransformAllFunction Normalize(
    std::vector<float> meanVector,
    std::vector<float> stdVector) {
  const af::array mean(1, 1, 3, 1, meanVector.data());
  const af::array std(1, 1, 3, 1, stdVector.data());
  return [mean, std](const std::vector<af::array>& in) {
    // Normalize Boxes
    auto boxes = in[BboxesIdx];

    if (!boxes.isempty()) {
      auto image = in[ImageIdx];
      auto w = float(image.dims(0));
      auto h = float(image.dims(1));

      boxes = xyxy2cxcywh(boxes);
      const std::vector<float> ratioVector = {w, h, w, h};
      af::array ratioArray = af::array(4, ratioVector.data());
      boxes = af::batchFunc(boxes, ratioArray, af::operator/);
    }
    // Normalize Image
    af::array image = in[ImageIdx].as(f32) / 255.f;
    image = af::batchFunc(image, mean, af::operator-);
    image = af::batchFunc(image, std, af::operator/);
    std::vector<af::array> outputs = {image,
                                      in[TargetSizeIdx],
                                      in[ImageIdIdx],
                                      in[OriginalSizeIdx],
                                      boxes,
                                      in[ClassesIdx]};
    return outputs;
  };
}

TransformAllFunction randomSelect(std::vector<TransformAllFunction> fns) {
  return [fns](const std::vector<af::array>& in) {
    TransformAllFunction randomFunc = fns[std::rand() % fns.size()];
    return randomFunc(in);
  };
};

TransformAllFunction randomSizeCrop(int minSize, int maxSize) {
  return [minSize, maxSize](const std::vector<af::array>& in) {
    const af::array& image = in[0];
    const int w = image.dims(0);
    const int h = image.dims(1);
    const int tw = randomInt(minSize, std::min(w, maxSize));
    const int th = randomInt(minSize, std::min(h, maxSize));
    const int x = std::rand() % (w - tw + 1);
    const int y = std::rand() % (h - th + 1);
    return crop(in, x, y, tw, th);
  };
};

TransformAllFunction randomResize(std::vector<int> sizes, int maxsize) {
  assert(sizes.size() > 0);
  auto resizeCoco = [sizes, maxsize](std::vector<af::array> in) {
    assert(in.size() == 6);
    assert(sizes.size() > 0);
    int randomIndex = rand() % sizes.size();
    int size = sizes[randomIndex];
    const af::array originalImage = in[0];
    return randomResize(in, size, maxsize);
  };
  return resizeCoco;
}

TransformAllFunction randomHorizontalFlip(float p) {
  return [p](const std::vector<af::array>& in) {
    if (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) > p) {
      return hflip(in);
    } else {
      return in;
    }
  };
}

TransformAllFunction compose(std::vector<TransformAllFunction> fns) {
  return [fns](const std::vector<af::array>& in) {
    std::vector<af::array> out = in;
    for (auto fn : fns) {
      out = fn(out);
    }
    return out;
  };
}

} // namespace vision
} // namespace pkg
} // namespace fl
