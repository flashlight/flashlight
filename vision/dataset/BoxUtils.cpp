#include "BoxUtils.h"

#include <arrayfire.h>
#include <assert.h>
#include <tuple>
#include <iostream>

//#include "flashlight/flashlight.h"
#include "flashlight/autograd/Functions.h"


namespace fl {
namespace cv {
namespace dataset {

//af::array cxcywh_to_xyxy(const af::array& bboxes) {
  //af::array transformed = af::constant(0, 5, bboxes.dims(1));
  //auto x_c = bboxes(0, af::span);
  //auto y_c = bboxes(1, af::span);
  //auto w = bboxes(2, af::span);
  //auto h = bboxes(3, af::span);
  //auto cls = bboxes(4, af::span);
  //transformed(0, af::span) = x_c - 0.5 * w;
  //transformed(1, af::span) = y_c - 0.5 * h;
  //transformed(2, af::span) = x_c + 0.5 * w;
  //transformed(3, af::span) = y_c + 0.5 * h;
  //transformed(4, af::span) = cls;
  //return transformed;
//}

fl::Variable cxcywh_to_xyxy(const Variable& bboxes) {
  af_print(bboxes.array());
  auto transformed = Variable(af::constant(0, 4, bboxes.dims(1)), true);
  auto x_c = bboxes.row(0);
  auto y_c = bboxes.row(1);
  auto w = bboxes.row(2);
  auto h = bboxes.row(3);
  transformed = fl::concatenate( { 
      x_c - 0.5 * w, 
      y_c - 0.5 * h,
      x_c + 0.5 * w,
      y_c + 0.5 * h
  }, 0);
  return transformed;
}

//af::array xyxy_to_cxcywh(const af::array& bboxes) {
  //af::array transformed = af::constant(0, 5, bboxes.dims(1));
  //auto x0 = bboxes(0, af::span);
  //auto y0 = bboxes(1, af::span);
  //auto x1 = bboxes(2, af::span);
  //auto y1 = bboxes(3, af::span);
  //auto cls = bboxes(4, af::span);
  //transformed(0, af::span) = (x0 + x1) / 2.f ;
  //transformed(1, af::span) = (y0 + y1) / 2.f;
  //transformed(2, af::span) = (x1 - x0);
  //transformed(3, af::span) = (y1 - y0);
  //transformed(4, af::span) = cls;
  //return transformed;
//}

//af::array xywh_to_cxcywh(const af::array& bboxes) {
  //af::array transformed = af::constant(0, 5, bboxes.dims(1));
  //auto x0 = bboxes(0, af::span);
  //auto y0 = bboxes(1, af::span);
  //auto w = bboxes(2, af::span);
  //auto h = bboxes(3, af::span);
  //auto cls = bboxes(4, af::span);
  //transformed(0, af::span) = x0 + (w / 2.f) ;
  //transformed(1, af::span) = y0 + (h / 2.f);
  //transformed(2, af::span) = w;
  //transformed(3, af::span) = h;
  //transformed(4, af::span) = cls;
  //return transformed;
//}

//af::array box_area(const af::array& bboxes) {
  //auto x0 = bboxes.row(0);
  //auto y0 = bboxes.row(1);
  //auto x1 = bboxes.row(2);
  //auto y1 = bboxes.row(3);
  //auto result = (x1 - x0) * (y1 - y0);
  //return result;
//}

// Given an [K, N, B, 1] for x
// and [K, M, B, 1]
// return [Function(K), N, M, B ] where the first two dimension are the result of applying fn
// to the catersan product of all Ns and Ms
//af::array cartesian(const af::array& x, const af::array& y, af::batchFunc_t fn) {
  //assert(y.dims(3) == 1);
  //assert(x.dims(3) == 1);
  //assert(x.dims(2) == y.dims(2));
  //af::dim4 y_dims = {y.dims(0), 1, y.dims(1), y.dims(2)};
  //auto y_mod = moddims(y, {y.dims(0), 1, y.dims(1), y.dims(2)});
  //auto x_mod = moddims(x, {x.dims(0), x.dims(1), 1, x.dims(2)});
  //x_mod = af::tile(x_mod, {1, 1, y.dims(1), 1});
  //y_mod = af::tile(y_mod, {1, x.dims(1), 1, 1});
  //return fn(x_mod, y_mod);
//}


//af::array squeeze(const af::array& x) {
  //af::dim4 dims = {1, 1, 1, 1};
  //int idx = 0;
  //for(int i = 0; i < 4; i++) {
    //if (x.dims(i) > 1) {
      //dims[idx] = x.dims(i); 
      //idx += 1;
    //}
  //}
  //return moddims(x, dims);
//}

//af::array flatten(const af::array& x, int start, int stop) {
  //auto dims = x.dims();
  //af::dim4 new_dims = { 1, 1, 1, 1};
  //int flattened_dims = 1;
  //for(int i = start; i <= stop; i++) {
    //flattened_dims = flattened_dims * dims[i];
  //}


  //[>
  //for(int i = 0; i < start; i++) {
    //new_dims[i] = dims[i];
  //}
  //new_dims[start] = flattened_dims;
  //for(int i = stop; i < (4 - stop); i++) {
    //new_dims[i] = dims[i + stop]
  //}
  //return new_dims;
  //*/

  //new_dims[0] = flattened_dims;
  //for(int i = 1; i < (4 - stop); i++) {
    //new_dims[i] = dims[i + stop];
  //}
  //return af::moddims(x, new_dims);
//};

fl::Variable flatten(const fl::Variable& x, int start, int stop) {
  auto dims = x.dims();
  af::dim4 new_dims = { 1, 1, 1, 1};
  int flattened_dims = 1;
  for(int i = start; i <= stop; i++) {
    flattened_dims = flattened_dims * dims[i];
  }
  new_dims[0] = flattened_dims;
  for(int i = 1; i < (4 - stop); i++) {
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
  af_print(area1.array());
  af_print(area2.array());
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), fl::max);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), min);
  af_print(lt.array());
  af_print(rb.array());
  auto wh = max((rb - lt), 0.0);
  af_print(wh.array());
  auto inter = wh.row(0) * wh.row(1);
  af_print(inter.array());
  auto uni = cartesian(area1, area2, fl::operator+) - inter;
  auto iou = inter / uni;
  iou = flatten(iou, 0, 1);
  uni = flatten(uni, 0, 1);
  return std::tie(iou, uni);
}

fl::Variable generalized_box_iou(const fl::Variable& bboxes1, const fl::Variable& bboxes2) {
  // Make sure all boxes are properly formed
  af_print(bboxes1.array());
  af_print(bboxes2.array());
  assert(af::count(allTrue(bboxes1.array().rows(2, 3) >= bboxes1.array().rows(0, 1))).scalar<uint32_t>());
  assert(af::count(allTrue(bboxes2.array().rows(2, 3) >= bboxes2.array().rows(0, 1))).scalar<uint32_t>());

  Variable iou, uni;
  std::tie(iou, uni) = box_iou(bboxes1, bboxes2);
  af_print(iou.array());
  af_print(uni.array());
  auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), min);
  auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), max);
  af_print(lt.array());
  af_print(rb.array());
  auto wh = max((rb - lt), 0.0);
  auto area = wh.row(0) * wh.row(1);
  area = flatten(area, 0, 1);
  return iou - (area - uni) / area;
}

// bboxes1 5, N
// bboxes2 5, M
// Expect [x1, y1, x2, y2]
//std::tuple<af::array, af::array> box_iou(const af::array& bboxes1, const af::array& bboxes2) {

  //auto area1 = box_area(bboxes1); // [N]
  //auto area2 = box_area(bboxes2); // [M]
  //auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), af::max);
  //auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), af::min);
  //auto wh = af::max((rb - lt), 0.0);
  //auto inter = wh.row(0) * wh.row(1);
  //auto uni = cartesian(area1, area2, af::operator+) - inter;
  //auto iou = inter / uni;
  //iou = flatten(iou, 0, 1);
  //uni = flatten(uni, 0, 1);
  //return std::tie(iou, uni);
//}

//af::array generalized_box_iou(const af::array& bboxes1, const af::array& bboxes2) {
  //// Make sure all boxes are properly formed
  //assert(af::count(allTrue(bboxes1.rows(2, 3) >= bboxes1.rows(0, 1))).scalar<uint32_t>());
  //assert(af::count(allTrue(bboxes2.rows(2, 3) >= bboxes2.rows(0, 1))).scalar<uint32_t>());

  //af::array iou, uni;
  //std::tie(iou, uni) = box_iou(bboxes1, bboxes2);
  //auto lt = cartesian(bboxes1.rows(0, 1), bboxes2.rows(0, 1), af::min);
  //auto rb = cartesian(bboxes1.rows(2, 3), bboxes2.rows(2, 3), af::max);
  //auto wh = af::max((rb - lt), 0.0);
  //auto area = wh.row(0) * wh.row(1);
  //af_print(iou);
  //af_print(area);
  //area = flatten(area, 0, 1);
  //af_print(area);
  //return iou - (area - uni) / area;
//}

Variable l1_loss(const Variable& input, const Variable& target) {
  return abs(input - target);
}

} // namespace dataset
} // namespace cv
} // namespace fl
