/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/CocoTransforms.h"

#include <cassert>

#include "flashlight/fl/tensor/Index.h"
#include "flashlight/pkg/vision/dataset/BoxUtils.h"
#include "flashlight/pkg/vision/dataset/TransformAllDataset.h"
#include "flashlight/pkg/vision/dataset/Transforms.h"
#include "flashlight/pkg/vision/tensor/VisionOps.h"

namespace {

int randomInt(int min, int max) {
  return std::rand() % (max - min + 1) + min;
}
} // namespace

namespace fl::pkg::vision {

std::vector<Tensor>
crop(const std::vector<Tensor>& in, int x, int y, int tw, int th) {
  const Tensor& image = in[ImageIdx];
  const Tensor croppedImage = fl::pkg::vision::crop(image, x, y, tw, th);

  const Tensor& boxes = in[BboxesIdx];

  const std::vector<int> translateVector = {x, y, x, y};
  const std::vector<int> maxSizeVector = {tw, th};
  Tensor targetSize = Tensor::fromVector(maxSizeVector);

  const Tensor translateArray = Tensor::fromVector(translateVector);
  const Tensor maxSizeArray = Tensor::fromVector(maxSizeVector);

  Tensor croppedBoxes = boxes;
  Tensor labels = in[ClassesIdx];

  if (!croppedBoxes.isEmpty()) {
    croppedBoxes = croppedBoxes - translateArray;
    croppedBoxes = fl::reshape(croppedBoxes, {2, 2, boxes.dim(1)});
    croppedBoxes = fl::minimum(croppedBoxes, maxSizeArray);
    croppedBoxes = fl::maximum(croppedBoxes, 0.0);
    Tensor keep = fl::all(
        croppedBoxes(fl::span, fl::range(1, 2), fl::span) >
            croppedBoxes(fl::span, fl::range(0, 1), fl::span),
        {0});
    croppedBoxes = fl::reshape(croppedBoxes, {4, boxes.dim(1)});
    croppedBoxes = croppedBoxes(fl::span, keep);
    labels = labels(fl::span, keep);
  }
  return {
      croppedImage,
      targetSize,
      in[ImageIdIdx],
      in[OriginalSizeIdx],
      croppedBoxes,
      labels};
};

std::vector<Tensor> hflip(const std::vector<Tensor>& in) {
  Tensor image = in[ImageIdx];
  const int w = image.dim(0);
  image = image(fl::range(w - 1, -1, -1));

  Tensor bboxes = in[BboxesIdx];
  if (!bboxes.isEmpty()) {
    Tensor bboxes_flip = Tensor(bboxes.shape());
    bboxes_flip(0) = (bboxes(2) * -1) + w;
    bboxes_flip(1) = bboxes(1);
    bboxes_flip(2) = (bboxes(0) * -1) + w;
    bboxes_flip(3) = bboxes(3);
    bboxes = bboxes_flip;
  }
  return {
      image,
      in[TargetSizeIdx],
      in[ImageIdIdx],
      in[OriginalSizeIdx],
      bboxes,
      in[ClassesIdx]};
}

std::vector<Tensor> normalize(const std::vector<Tensor>& in) {
  auto boxes = in[BboxesIdx];

  if (!boxes.isEmpty()) {
    auto image = in[ImageIdx];
    auto w = float(image.dim(0));
    auto h = float(image.dim(1));

    boxes = xyxy2cxcywh(boxes);
    const std::vector<float> ratioVector = {w, h, w, h};
    Tensor ratioArray = Tensor::fromVector(ratioVector);
    boxes = boxes / ratioArray;
  }
  return {
      in[ImageIdx],
      in[TargetSizeIdx],
      in[ImageIdIdx],
      in[OriginalSizeIdx],
      boxes,
      in[ClassesIdx]};
}

std::vector<Tensor>
randomResize(std::vector<Tensor> inputs, int size, int maxsize) {
  auto getSize = [](const Tensor& in, int size, int maxSize = 0) {
    int w = in.dim(0);
    int h = in.dim(1);
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

  Tensor image = inputs[ImageIdx];
  auto output_size = getSize(image, size, maxsize);
  const Shape originalDims = image.shape();
  Tensor resizedImage;
  resizedImage = fl::resize(
      image,
      {output_size.first, output_size.second},
      InterpolationMode::Bilinear);
  const Shape resizedDims = resizedImage.shape();

  Tensor boxes = inputs[BboxesIdx];
  if (!boxes.isEmpty()) {
    const float ratioWidth = float(resizedDims[0]) / float(originalDims[0]);
    const float ratioHeight = float(resizedDims[1]) / float(originalDims[1]);

    const std::vector<float> resizeVector = {
        ratioWidth, ratioHeight, ratioWidth, ratioHeight};
    Tensor resizedArray = Tensor::fromVector(resizeVector);
    boxes = boxes * resizedArray;
  }

  std::vector<long> imageSizeArray = {resizedImage.dim(1), resizedImage.dim(0)};
  Tensor sizeArray = Tensor::fromVector(imageSizeArray);
  return {
      resizedImage,
      sizeArray,
      inputs[ImageIdIdx],
      inputs[OriginalSizeIdx],
      boxes,
      inputs[ClassesIdx]};
}

TransformAllFunction Normalize(
    std::vector<float> meanVector,
    std::vector<float> stdVector) {
  const Tensor mean = Tensor::fromVector({1, 1, 3}, meanVector);
  const Tensor std = Tensor::fromVector({1, 1, 3}, stdVector);
  return [mean, std](const std::vector<Tensor>& in) {
    // Normalize Boxes
    auto boxes = in[BboxesIdx];

    if (!boxes.isEmpty()) {
      auto image = in[ImageIdx];
      auto w = float(image.dim(0));
      auto h = float(image.dim(1));

      boxes = xyxy2cxcywh(boxes);
      const std::vector<float> ratioVector = {w, h, w, h};
      Tensor ratioArray = Tensor::fromVector(ratioVector);
      boxes = boxes / ratioArray;
    }
    // Normalize Image
    Tensor image = in[ImageIdx].astype(fl::dtype::f32) / 255.f;
    image = image - mean;
    image = image / std;
    std::vector<Tensor> outputs = {
        image,
        in[TargetSizeIdx],
        in[ImageIdIdx],
        in[OriginalSizeIdx],
        boxes,
        in[ClassesIdx]};
    return outputs;
  };
}

TransformAllFunction randomSelect(std::vector<TransformAllFunction> fns) {
  return [fns](const std::vector<Tensor>& in) {
    TransformAllFunction randomFunc = fns[std::rand() % fns.size()];
    return randomFunc(in);
  };
};

TransformAllFunction randomSizeCrop(int minSize, int maxSize) {
  return [minSize, maxSize](const std::vector<Tensor>& in) {
    const Tensor& image = in[0];
    const int w = image.dim(0);
    const int h = image.dim(1);
    const int tw = randomInt(minSize, std::min(w, maxSize));
    const int th = randomInt(minSize, std::min(h, maxSize));
    const int x = std::rand() % (w - tw + 1);
    const int y = std::rand() % (h - th + 1);
    return crop(in, x, y, tw, th);
  };
};

TransformAllFunction randomResize(std::vector<int> sizes, int maxsize) {
  assert(!sizes.empty());
  auto resizeCoco = [sizes, maxsize](std::vector<Tensor> in) {
    assert(in.size() == 6);
    assert(!sizes.empty());
    int randomIndex = rand() % sizes.size();
    int size = sizes[randomIndex];
    const Tensor originalImage = in[0];
    return randomResize(in, size, maxsize);
  };
  return resizeCoco;
}

TransformAllFunction randomHorizontalFlip(float p) {
  return [p](const std::vector<Tensor>& in) {
    if (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) > p) {
      return hflip(in);
    } else {
      return in;
    }
  };
}

TransformAllFunction compose(std::vector<TransformAllFunction> fns) {
  return [fns](const std::vector<Tensor>& in) {
    std::vector<Tensor> out = in;
    for (const auto& fn : fns) {
      out = fn(out);
    }
    return out;
  };
}

} // namespace fl
