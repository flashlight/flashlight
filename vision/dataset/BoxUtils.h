#include <arrayfire.h>
#include <tuple>

#include "flashlight/flashlight.h"

namespace fl {
namespace cv {
namespace dataset {

af::array cxcywh_to_xyxy(const af::array& bboxes);

af::array xyxy_to_cxcywh(const af::array& bboxes);

af::array xywh_to_cxcywh(const af::array& bboxes);

af::array box_area(const af::array& bboxes);

af::array squeeze(const af::array& x);

af::array cartesian(const af::array& x, const af::array& y, af::batchFunc_t fn);

// with bboxes 1 of length N
// and bboxes 2 of length M
// returns and N * M matrix where A(i, j) is cost of bboxes2[i], and bboxes2[j]
std::tuple<af::array, af::array> box_iou(const af::array& bboxes1, const af::array& bboxes2);

af::array generalized_box_iou(const af::array& bboxes1, const af::array& bboxes2);

Variable generalized_box_iou(const Variable& bboxes1, const Variable& bboxes2);

} // namespace dataset
} // namespace cv
} // namespace fl
