#pragma once

#include <arrayfire.h>
#include <tuple>

#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace app {
namespace objdet {

//af::array cxcywh_to_xyxy(const af::array& bboxes);

fl::Variable cxcywh_to_xyxy(const fl::Variable& bboxes);

af::array xyxy_to_cxcywh(const af::array& bboxes);

//af::array xywh_to_cxcywh(const af::array& bboxes);

//af::array box_area(const af::array& bboxes);

//af::array squeeze(const af::array& x);

//af::array cartesian(const af::array& x, const af::array& y, af::batchFunc_t fn);


typedef Variable(* batchFuncVar_t) (const Variable& lhs, const Variable& rhs);
fl::Variable cartesian(const fl::Variable& x, const fl::Variable& y, batchFuncVar_t fn);

//af::array flatten(const af::array& x, int start, int stop);

fl::Variable flatten(const fl::Variable& x, int start, int stop);

// with bboxes 1 of length N
// and bboxes 2 of length M
// returns and N * M matrix where A(i, j) is cost of bboxes2[i], and bboxes2[j]
//std::tuple<af::array, af::array> box_iou(const af::array& bboxes1, const af::array& bboxes2);

//af::array generalized_box_iou(const af::array& bboxes1, const af::array& bboxes2);

Variable generalized_box_iou(const Variable& bboxes1, const Variable& bboxes2);

Variable l1_loss(const Variable& input, const Variable& target);

} // namespace objdet
} // namespace app
} // namespace fl
