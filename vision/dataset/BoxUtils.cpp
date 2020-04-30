#include "BoxUtils.h"

#include <arrayfire.h>
#include <assert.h>

namespace fl {
namespace cv {
namespace dataset {

af::array cxcywh_to_xyxy(const af::array& bboxes) {
  af::array transformed = af::constant(0, 5, bboxes.dims(1));
  auto x_c = bboxes(0, af::span);
  auto y_c = bboxes(1, af::span);
  auto w = bboxes(2, af::span);
  auto h = bboxes(3, af::span);
  auto cls = bboxes(4, af::span);
  transformed(0, af::span) = x_c - 0.5 * w;
  transformed(1, af::span) = y_c - 0.5 * h;
  transformed(2, af::span) = x_c + 0.5 * w;
  transformed(3, af::span) = y_c + 0.5 * h;
  transformed(4, af::span) = cls;
  return transformed;
}

af::array xyxy_to_cxcywh(const af::array& bboxes) {
  af::array transformed = af::constant(0, 5, bboxes.dims(1));
  auto x0 = bboxes(0, af::span);
  auto y0 = bboxes(1, af::span);
  auto x1 = bboxes(2, af::span);
  auto y1 = bboxes(3, af::span);
  auto cls = bboxes(4, af::span);
  transformed(0, af::span) = (x0 + x1) / 2.f ;
  transformed(1, af::span) = (y0 + y1) / 2.f;
  transformed(2, af::span) = (x1 - x0);
  transformed(3, af::span) = (y1 - y0);
  transformed(4, af::span) = cls;
  return transformed;
}

af::array xywh_to_cxcywh(const af::array& bboxes) {
  af::array transformed = af::constant(0, 5, bboxes.dims(1));
  auto x0 = bboxes(0, af::span);
  auto y0 = bboxes(1, af::span);
  auto w = bboxes(2, af::span);
  auto h = bboxes(3, af::span);
  auto cls = bboxes(4, af::span);
  transformed(0, af::span) = x0 + (w / 2.f) ;
  transformed(1, af::span) = y0 + (h / 2.f);
  transformed(2, af::span) = w;
  transformed(3, af::span) = h;
  transformed(4, af::span) = cls;
  return transformed;
}

af::array box_area(const af::array& bboxes) {
  auto x0 = bboxes.row(0);
  auto y0 = bboxes.row(1);
  auto x1 = bboxes.row(2);
  auto y1 = bboxes.row(3);
  auto result = (x1 - x0) * (y1 - y0);
  return result;
}

// Given an [K, N, K, 1] for x
// and [K, M, K, 1]
// return [N, M, K, 1] where the first two dimension are the result of applying fn
// to the catersan product of all Ns and Ms
af::array cartesian(const af::array& x, const af::array& y, af::batchFunc_t fn) {
  assert(y.dims(3) == 1);
  assert(x.dims(3) == 1);
  af::dim4 dims = {y.dims(0), 1, y.dims(1), y.dims(2)};
  auto y_mod = moddims(y, dims);
  return batchFunc(x, y_mod, fn);
}

af::array squeeze(const af::array& x) {
  af::dim4 dims = {1, 1, 1, 1};
  int idx = 0;
  for(int i = 0; i < 4; i++) {
    if (x.dims(i) > 1) {
      dims[idx] = x.dims(i); 
      idx += 1;
    }
  }
  return moddims(x, dims);
}

// bboxes1 5, N
// bboxes2 5, M
// Expect [x1, y1, x2, y2]
af::array box_iou(const af::array& bboxes1, const af::array& bboxes2) {

  af::array area1 = box_area(bboxes1); // [N]
  af::array area2 = box_area(bboxes2); // [M]
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), af::max);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), af::min);
  auto wh = af::max((rb - lt), 0.0);
  auto inter = wh.row(0) * wh.row(1);
  auto uni = cartesian(area1, area2, af::operator+) - inter;
  return squeeze(inter / uni);
}

} // namespace dataset
} // namespace cv
} // namespace fl
