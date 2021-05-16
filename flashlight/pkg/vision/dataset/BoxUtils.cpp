/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/pkg/vision/dataset/BoxUtils.h"

#include <arrayfire.h>
#include <assert.h>
#include <tuple>

#include "flashlight/fl/autograd/Functions.h"

namespace fl {
namespace app {
namespace objdet {

af::array cxcywh2xyxy(const af::array& bboxes) {
  auto transformed = af::constant(0, 4, bboxes.dims(1));
  auto xc = bboxes.row(0);
  auto yc = bboxes.row(1);
  auto w = bboxes.row(2);
  auto h = bboxes.row(3);
  transformed =
      af::join(0, xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h);
  return transformed;
}

fl::Variable cxcywh2xyxy(const Variable& bboxes) {
  auto transformed = Variable(af::constant(0, 4, bboxes.dims(1)), true);
  auto xc = bboxes.row(0);
  auto yc = bboxes.row(1);
  auto w = bboxes.row(2);
  auto h = bboxes.row(3);
  transformed = fl::concatenate(
      {xc - 0.5 * w, yc - 0.5 * h, xc + 0.5 * w, yc + 0.5 * h}, 0);
  return transformed;
}

af::array xyxy2cxcywh(const af::array& bboxes) {
  auto x0 = bboxes.row(0);
  auto y0 = bboxes.row(1);
  auto x1 = bboxes.row(2);
  auto y1 = bboxes.row(3);
  af::array result =
      af::join(0, (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0));
  return result;
}

af::array flatten(const af::array& x, int start, int stop) {
  auto dims = x.dims();
  af::dim4 newDims = {1, 1, 1, 1};
  int flattenedDims = 1;
  for (int i = start; i <= stop; i++) {
    flattenedDims = flattenedDims * dims[i];
  }
  for (int i = 0; i < start; i++) {
    newDims[i] = dims[i];
  }
  newDims[start] = flattenedDims;
  for (int i = start + 1; i < (4 - stop); i++) {
    newDims[i] = dims[i + stop];
  }
  return af::moddims(x, newDims);
};

fl::Variable flatten(const fl::Variable& x, int start, int stop) {
  auto dims = x.dims();
  af::dim4 newDims = {1, 1, 1, 1};
  int flattenedDims = 1;
  for (int i = start; i <= stop; i++) {
    flattenedDims = flattenedDims * dims[i];
  }
  for (int i = 0; i < start; i++) {
    newDims[i] = dims[i];
  }
  newDims[start] = flattenedDims;
  for (int i = start + 1; i < (4 - stop); i++) {
    newDims[i] = dims[i + stop];
  }
  return fl::moddims(x, newDims);
};

af::array boxArea(const af::array& bboxes) {
  auto x0 = bboxes.row(0);
  auto y0 = bboxes.row(1);
  auto x1 = bboxes.row(2);
  auto y1 = bboxes.row(3);
  auto result = (x1 - x0) * (y1 - y0);
  return result;
}

fl::Variable boxArea(const fl::Variable& bboxes) {
  auto x0 = bboxes.row(0);
  auto y0 = bboxes.row(1);
  auto x1 = bboxes.row(2);
  auto y1 = bboxes.row(3);
  auto result = (x1 - x0) * (y1 - y0);
  return result;
}

//using batchFuncVar_t = std::function<Variable(const Variable& lhs, const Variable& rhs)>;

Variable cartesian(const Variable& x, const Variable& y, batchFuncVar_t fn) {
  assert(y.dims(3) == 1);
  assert(x.dims(3) == 1);
  assert(x.dims(2) == y.dims(2));
  af::dim4 yDims = {y.dims(0), 1, y.dims(1), y.dims(2)};
  auto yMod = moddims(y, {y.dims(0), 1, y.dims(1), y.dims(2)});
  auto xMod = moddims(x, {x.dims(0), x.dims(1), 1, x.dims(2)});
  af::dim4 outputDims = {x.dims(0), x.dims(1), y.dims(1), x.dims(2)};
  xMod = tileAs(xMod, outputDims);
  yMod = tileAs(yMod, outputDims);
  return fn(xMod, yMod);
}

af::array cartesian(const af::array& x, const af::array& y, batchFuncArr_t fn) {
  assert(y.dims(3) == 1);
  assert(x.dims(3) == 1);
  assert(x.dims(2) == y.dims(2));
  af::dim4 yDims = {y.dims(0), 1, y.dims(1), y.dims(2)};
  auto yMod = af::moddims(y, {y.dims(0), 1, y.dims(1), y.dims(2)});
  auto xMod = af::moddims(x, {x.dims(0), x.dims(1), 1, x.dims(2)});
  af::dim4 outputDims = {x.dims(0), x.dims(1), y.dims(1), x.dims(2)};
  xMod = detail::tileAs(xMod, outputDims);
  yMod = detail::tileAs(yMod, outputDims);
  return fn(xMod, yMod);
}

std::tuple<af::array, af::array> boxIou(
    const af::array& bboxes1,
    const af::array& bboxes2) {
  auto area1 = boxArea(bboxes1);
  auto area2 = boxArea(bboxes2);
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), af::max);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), af::min);
  auto wh = max((rb - lt), 0.0);
  auto inter = wh.row(0) * wh.row(1);
  auto uni = cartesian(area1, area2, af::operator+) - inter;
  auto iou = inter / uni;
  iou = flatten(iou, 0, 1);
  uni = flatten(uni, 0, 1);
  return std::tie(iou, uni);
}

std::tuple<fl::Variable, fl::Variable> boxIou(
    const fl::Variable& bboxes1,
    const fl::Variable& bboxes2) {
  auto area1 = boxArea(bboxes1);
  auto area2 = boxArea(bboxes2);
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), fl::max);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), min);
  auto wh = max((rb - lt), 0.0);
  auto inter = wh.row(0) * wh.row(1);
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
  assert(af::count(
             allTrue(bboxes1.array().rows(2, 3) >= bboxes1.array().rows(0, 1)))
             .scalar<uint32_t>());
  assert(af::count(
             allTrue(bboxes2.array().rows(2, 3) >= bboxes2.array().rows(0, 1)))
             .scalar<uint32_t>());

  Variable iou, uni;
  std::tie(iou, uni) = boxIou(bboxes1, bboxes2);
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), min);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), max);
  auto wh = max((rb - lt), 0.0);
  auto area = wh.row(0) * wh.row(1);
  area = flatten(area, 0, 1);
  return iou - (area - uni) / area;
}

af::array generalizedBoxIou(
    const af::array& bboxes1,
    const af::array& bboxes2) {
  // Make sure all boxes are properly formed
  assert(af::count(allTrue(bboxes1.rows(2, 3) >= bboxes1.rows(0, 1)))
             .scalar<uint32_t>());
  assert(af::count(allTrue(bboxes2.rows(2, 3) >= bboxes2.rows(0, 1)))
             .scalar<uint32_t>());

  af::array iou, uni;
  std::tie(iou, uni) = boxIou(bboxes1, bboxes2);
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), af::min);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), af::max);
  auto wh = max((rb - lt), 0.0);
  auto area = wh.row(0) * wh.row(1);
  area = flatten(area, 0, 1);
  return iou - (area - uni) / area;
}

Variable l1Loss(const Variable& input, const Variable& target) {
  return flatten(fl::sum(fl::abs(input - target), {0}), 0, 1);
}

} // namespace objdet
} // namespace app
} // namespace fl
