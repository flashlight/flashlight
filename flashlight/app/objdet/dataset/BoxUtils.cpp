#include "BoxUtils.h"

#include <arrayfire.h>
#include <assert.h>
#include <tuple>
#include <iostream>

#include "flashlight/fl/autograd/Functions.h"

namespace fl {
namespace app {
namespace objdet {

af::array xyxy_to_cxcywh(const af::array& bboxes) {
  auto x0 = bboxes.row(0);
  auto y0 = bboxes.row(1);
  auto x1 = bboxes.row(2);
  auto y1 = bboxes.row(3);
  af::array result =  af::join(0, (x0 + x1) / 2, (y0 + y1) / 2,
      (x1 - x0), (y1 - y0));
  return result;
}

fl::Variable cxcywh_to_xyxy(const Variable& bboxes) {
  auto transformed = Variable(af::constant(0, 4, bboxes.dims(1)), true);
  auto x_c = bboxes.row(0);
  auto y_c = bboxes.row(1);
  auto w = bboxes.row(2);
  auto h = bboxes.row(3);
  transformed = fl::concatenate({
      x_c - 0.5 * w,
      y_c - 0.5 * h,
      x_c + 0.5 * w,
      y_c + 0.5 * h
  }, 0);
  return transformed;
}

fl::Variable flatten(const fl::Variable& x, int start, int stop) {
  auto dims = x.dims();
  af::dim4 new_dims = { 1, 1, 1, 1};
  int flattened_dims = 1;
  for(int i = start; i <= stop; i++) {
    flattened_dims = flattened_dims * dims[i];
  }
  for(int i = 0; i < start; i++) {
    new_dims[i] = dims[i];
  }
  new_dims[start] = flattened_dims;
  for(int i = start + 1; i < (4 - stop); i++) {
    new_dims[i] = dims[i + stop];
  }
  return fl::moddims(x, new_dims);
};

fl::Variable box_area(const fl::Variable& bboxes) {
  auto x0 = bboxes.row(0);
  auto y0 = bboxes.row(1);
  auto x1 = bboxes.row(2);
  auto y1 = bboxes.row(3);
  auto result = (x1 - x0) * (y1 - y0);
  return result;
}

typedef Variable(* batchFuncVar_t) (const Variable& lhs, const Variable& rhs);


Variable cartesian(
    const Variable& x,
    const Variable& y,
    batchFuncVar_t fn) {
  assert(y.dims(3) == 1);
  assert(x.dims(3) == 1);
  assert(x.dims(2) == y.dims(2));
  af::dim4 y_dims = {y.dims(0), 1, y.dims(1), y.dims(2)};
  auto y_mod = moddims(y, {y.dims(0), 1, y.dims(1), y.dims(2)});
  auto x_mod = moddims(x, {x.dims(0), x.dims(1), 1, x.dims(2)});
  af::dim4 output_dims = {x.dims(0), x.dims(1), y.dims(1), x.dims(2)};
  x_mod = tileAs(x_mod, output_dims);
  y_mod = tileAs(y_mod, output_dims);
  return fn(x_mod, y_mod);
}

std::tuple<fl::Variable, fl::Variable> box_iou(
    const fl::Variable& bboxes1,
    const fl::Variable& bboxes2) {
  auto area1 = box_area(bboxes1);
  auto area2 = box_area(bboxes2);
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

fl::Variable generalized_box_iou(const fl::Variable& bboxes1, const fl::Variable& bboxes2) {
  // Make sure all boxes are properly formed
  assert(af::count(allTrue(bboxes1.array().rows(2, 3) >= bboxes1.array().rows(0, 1))).scalar<uint32_t>());
  assert(af::count(allTrue(bboxes2.array().rows(2, 3) >= bboxes2.array().rows(0, 1))).scalar<uint32_t>());

  Variable iou, uni;
  std::tie(iou, uni) = box_iou(bboxes1, bboxes2);
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), min);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), max);
  auto wh = max((rb - lt), 0.0);
  auto area = wh.row(0) * wh.row(1);
  area = flatten(area, 0, 1);
  return iou - (area - uni) / area;
}

Variable l1_loss(const Variable& input, const Variable& target) {
  return flatten(sum(abs(input - target), {0} ), 0, 1);
}

} // namespace objdet
} // namespace app
} // namespace fl
