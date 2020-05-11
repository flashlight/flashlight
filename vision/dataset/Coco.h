#pragma once

#include "vision/dataset/ImageDataset.h"
#include "vision/dataset/Transforms.h"
#include "flashlight/dataset/datasets.h"

namespace fl {
namespace cv {
namespace dataset {

std::shared_ptr<Dataset> coco(
    const std::string& list_file,
    std::vector<ImageTransform>& transformfns);

//af::array cxcywh_to_xyxy(const af::array& bboxes);

//af::array xyxy_to_cxcywh(const af::array& bboxes);

//af::array xywh_to_cxcywh(const af::array& bboxes);

//af::array box_iou(const af::array& bboxes1, const af::array& bboxes2);

} // namespace dataset
} // namespace cv
} // namespace flashlight
