/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/BoxUtils.h"

#include <cassert>
#include <stdexcept>
#include <tuple>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl::pkg::vision {

Tensor cxcywh2xyxy(const Tensor& bboxes) {
  auto xc = bboxes(fl::range(0, 1));
  auto yc = bboxes(fl::range(1, 2));
  auto w = bboxes(fl::range(2, 3));
  auto h = bboxes(fl::range(3, 4));

  return fl::concatenate(
      0, xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h);
}

fl::Variable cxcywh2xyxy(const Variable& bboxes) {
  auto xc = bboxes(fl::range(0, 1));
  auto yc = bboxes(fl::range(1, 2));
  auto w = bboxes(fl::range(2, 3));
  auto h = bboxes(fl::range(3, 4));

  return fl::concatenate(
      {xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h}, 0);
}

Tensor xyxy2cxcywh(const Tensor& bboxes) {
  auto x0 = bboxes(fl::range(0, 1));
  auto y0 = bboxes(fl::range(1, 2));
  auto x1 = bboxes(fl::range(2, 3));
  auto y1 = bboxes(fl::range(3, 4));
  Tensor result =
      fl::concatenate(0, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0));
  return result;
}

Tensor flatten(const Tensor& x, int start, int stop) {
  auto dims = x.shape();
  Shape newDims(std::vector<Dim>(x.ndim(), 1));
  int flattenedDims = 1;
  for (int i = start; i <= stop; i++) {
    flattenedDims = flattenedDims * dims[i];
  }
  for (int i = 0; i < start; i++) {
    newDims[i] = dims[i];
  }
  newDims[start] = flattenedDims;
  for (int i = start + 1; i < (x.ndim() - stop); i++) {
    newDims[i] = dims[i + stop];
  }
  return fl::reshape(x, newDims);
};

fl::Variable flatten(const fl::Variable& x, int start, int stop) {
  unsigned n = x.ndim();
  auto dims = x.shape();
  Shape newDims(std::vector<Dim>(n, 1));
  int flattenedDims = 1;
  for (int i = start; i <= stop; i++) {
    flattenedDims = flattenedDims * dims[i];
  }
  for (int i = 0; i < start; i++) {
    newDims[i] = dims[i];
  }
  newDims[start] = flattenedDims;
  for (int i = start + 1; i < (n - stop); i++) {
    newDims[i] = dims[i + stop];
  }
  return moddims(x, newDims);
};

Tensor boxArea(const Tensor& bboxes) {
  auto x0 = bboxes(fl::range(0, 1));
  auto y0 = bboxes(fl::range(1, 2));
  auto x1 = bboxes(fl::range(2, 3));
  auto y1 = bboxes(fl::range(3, 4));
  auto result = (x1 - x0) * (y1 - y0);
  return result;
}

fl::Variable boxArea(const fl::Variable& bboxes) {
  auto x0 = bboxes(fl::range(0, 1));
  auto y0 = bboxes(fl::range(1, 2));
  auto x1 = bboxes(fl::range(2, 3));
  auto y1 = bboxes(fl::range(3, 4));
  auto result = (x1 - x0) * (y1 - y0);
  return result;
}

Variable cartesian(const Variable& x, const Variable& y, batchFuncVar_t fn) {
  if (x.ndim() != 3 || y.ndim() != 3) {
    throw std::invalid_argument(
        "vision::cartesian - x and y inputs must have 3 dimensions");
  }
  assert(x.dim(2) == y.dim(2));
  Shape yDims = {y.dim(0), 1, y.dim(1), y.dim(2)};
  auto yMod = moddims(y, {y.dim(0), 1, y.dim(1), y.dim(2)});
  auto xMod = moddims(x, {x.dim(0), x.dim(1), 1, x.dim(2)});
  Shape outputDims = {x.dim(0), x.dim(1), y.dim(1), x.dim(2)};
  xMod = tileAs(xMod, outputDims);
  yMod = tileAs(yMod, outputDims);

  auto out = fn(xMod, yMod);
  return out;
}

Tensor cartesian(const Tensor& x, const Tensor& y, batchFuncArr_t fn) {
  if (x.ndim() != 3 || y.ndim() != 3) {
    throw std::invalid_argument(
        "vision::cartesian - x and y inputs must have 3 dimensions");
  }
  assert(x.dim(2) == y.dim(2));
  Shape yDims = {y.dim(0), 1, y.dim(1), y.dim(2)};
  auto yMod = fl::reshape(y, {y.dim(0), 1, y.dim(1), y.dim(2)});
  auto xMod = fl::reshape(x, {x.dim(0), x.dim(1), 1, x.dim(2)});
  Shape outputDims = {x.dim(0), x.dim(1), y.dim(1), x.dim(2)};
  xMod = detail::tileAs(xMod, outputDims);
  yMod = detail::tileAs(yMod, outputDims);
  return fn(xMod, yMod);
}

std::tuple<Tensor, Tensor> boxIou(
    const Tensor& bboxes1,
    const Tensor& bboxes2) {
  if (bboxes1.ndim() != 3 || bboxes2.ndim() != 3) {
    throw std::invalid_argument(
        "vision::boxIou - bbox inputs must be of shape "
        "[4, N, B, ...] and [4, M, B, ...]");
  }
  auto area1 = boxArea(bboxes1);
  auto area2 = boxArea(bboxes2);
  auto lt = cartesian(
      bboxes1(fl::range(0, 2)), bboxes2(fl::range(0, 2)), fl::maximum);
  auto rb = cartesian(
      bboxes1(fl::range(2, 4)), bboxes2(fl::range(2, 4)), fl::minimum);
  auto wh = fl::maximum((rb - lt), 0.0);
  auto inter = wh(fl::range(0, 1)) * wh(fl::range(1, 2));
  auto uni = cartesian(area1, area2, fl::operator+) - inter;
  auto iou = inter / uni;
  iou = flatten(iou, 0, 1);
  uni = flatten(uni, 0, 1);
  return std::tie(iou, uni);
}

std::tuple<fl::Variable, fl::Variable> boxIou(
    const fl::Variable& bboxes1,
    const fl::Variable& bboxes2) {
  if (bboxes1.ndim() != 3 || bboxes2.ndim() != 3) {
    std::stringstream ss;
    ss << "vision::boxIou - bbox inputs must be of shape "
          "[4, N, B] and [4, M, B]. Got boxes with dimensions "
       << bboxes1.shape() << " and " << bboxes2.shape();
    throw std::invalid_argument(ss.str());
  }
  auto area1 = boxArea(bboxes1);
  auto area2 = boxArea(bboxes2);
  auto lt =
      cartesian(bboxes1(fl::range(0, 2)), bboxes2(fl::range(0, 2)), fl::max);
  auto rb = cartesian(bboxes1(fl::range(2, 4)), bboxes2(fl::range(2, 4)), min);
  auto wh = max((rb - lt), 0.0);
  auto inter = wh(fl::range(0, 1)) * wh(fl::range(1, 2));
  auto uni = cartesian(area1, area2, fl::operator+) - inter;
  auto iou = inter / uni;
  iou = flatten(iou, 0, 1);
  uni = flatten(uni, 0, 1);
  return std::tie(iou, uni);
}

fl::Variable generalizedBoxIou(
    const fl::Variable& bboxes1,
    const fl::Variable& bboxes2) {
  // Make sure all boxes are properly formed
  assert(fl::countNonzero(fl::all(
                              bboxes1.tensor()(fl::range(2, 4)) >=
                              bboxes1.tensor()(fl::range(0, 2))))
             .scalar<uint32_t>());

  assert(fl::countNonzero(fl::all(
                              bboxes2.tensor()(fl::range(2, 4)) >=
                              bboxes2.tensor()(fl::range(0, 2))))
             .scalar<uint32_t>());

  Variable iou, uni;
  std::tie(iou, uni) = boxIou(bboxes1, bboxes2);
  auto lt = cartesian(bboxes1(fl::range(0, 2)), bboxes2(fl::range(0, 2)), min);
  auto rb = cartesian(bboxes1(fl::range(2, 4)), bboxes2(fl::range(2, 4)), max);
  auto wh = max((rb - lt), 0.0);
  auto area = wh(fl::range(0, 1)) * wh(fl::range(1, 2));
  area = flatten(area, 0, 1);
  return iou - (area - uni) / area;
}

Tensor generalizedBoxIou(const Tensor& bboxes1, const Tensor& bboxes2) {
  // Make sure all boxes are properly formed
  assert(fl::countNonzero(
             fl::all(bboxes1(fl::range(2, 4)) >= bboxes1(fl::range(0, 2))))
             .scalar<uint32_t>());
  assert(fl::countNonzero(
             fl::all(bboxes2(fl::range(2, 4)) >= bboxes2(fl::range(0, 2))))
             .scalar<uint32_t>());

  Tensor iou, uni;
  std::tie(iou, uni) = boxIou(bboxes1, bboxes2);
  auto lt = cartesian(
      bboxes1(fl::range(0, 2)), bboxes2(fl::range(0, 2)), fl::minimum);
  auto rb = cartesian(
      bboxes1(fl::range(2, 4)), bboxes2(fl::range(2, 4)), fl::maximum);
  auto wh = fl::maximum((rb - lt), 0.0);
  auto area = wh(fl::range(0, 1)) * wh(fl::range(1, 2));
  area = flatten(area, 0, 1);
  return iou - (area - uni) / area;
}

Variable l1Loss(const Variable& input, const Variable& target) {
  return flatten(fl::sum(fl::abs(input - target), {0}), 0, 1);
}

} // namespace fl
