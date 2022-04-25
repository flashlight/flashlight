/*
 * Copyright (c) Facebook, Inc. and its affiliates.  * All rights reserved.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <tuple>

#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace pkg {
namespace vision {

using batchFuncVar_t = Variable (*)(const Variable &, const Variable &);

using batchFuncArr_t = Tensor (*)(const Tensor &, const Tensor &);

/**
 * Converts bounding box coordinates from center (x, y) coordinate, with width
 * and height, to bottom left (x1, y1) top right (x2, y2) coordinates.
 * @param bboxes an Tensor with shape \f$[4, N]\f$ where N is the number of
 * boxes
 * @return a `Tensor` with transformed bboxes of same shape
 */
Tensor cxcywh2xyxy(const Tensor& bboxes);

/**
 * Converts bounding box coordinates from center (x, y) coordinate, with width
 * and height, to bottom left (x1, y1) top right (x2, y2) coordinates.
 * @param bboxes an fl::Variable with shape \f$[4, N]\f$ where N is the number
 * of boxes
 * @return a `fl::Variable` with transformed bboxes of same shape
 */
fl::Variable cxcywh2xyxy(const fl::Variable& bboxes);

/**
 * Converts bounding box coordinates from  bottom left (x1, y1) top right
 * (x2, y2) coordinates * to a center (x, y) coordinate, with width * and
 * height.
 * @param bboxes an Tensor with shape \f$[4, N]\f$ where N is the number of
 * boxes
 * @return a `Tensor` with transformed bboxes of same shape
 */
Tensor xyxy2cxcywh(const Tensor& bboxes);

/**
 * A generalized function for getting the "cartesian" product of a function
 * between two arrays. This is useful for bounding box functions when we have
 * an array of [4 X N] and [4 X M] and we want to apply an arbitrary function
 * to get an array of [ N X M ], * where each entry represents to pairwise
 * value between the two input arrays
 * @param x a fl::Variable of shape [ X x N x K x 1 ] and
 * @param y a fl::Variable of shape [ X x M x K x 1 ]
 * @return a fl::Variable of shape [ X x N X M X K ]
 *
 */
fl::Variable
cartesian(const fl::Variable& x, const fl::Variable& y, batchFuncVar_t fn);

/**
 * A generalized function for getting the "cartesian" product of a function
 * between two arrays. This is useful for bounding box functions when we have
 * an array of [4 X N X B] and [4 X M X B] and we want to apply an arbitrary
 * function to get an array of [ N X M ], * where each entry represents to
 * pairwise value between the two input arrays. (To get IOU between all boxes)
 * @param x a Tensor of shape [ X x N x K x 1 ] and
 * @param y a Tensor of shape [ X x M x K x 1 ]
 * @return a Tensor of shape [ X x N X M X K ]
 *
 */
Tensor cartesian(const Tensor& x, const Tensor& y, batchFuncArr_t fn);

/**
 * Flattens dimension between start and stop in an Tensor
 * @param x an Tensor of an arbitrary shape
 * @param start an int, the starting dimension to flatten
 * @param stop an int, the end dimension to flatten
 * @return an Tensor with collasped dimensions
 */
Tensor flatten(const Tensor& x, int start, int stop);

/**
 * Flattens dimension between start and stop in an Tensor
 * @param x an fl::array of an arbitrary shape
 * @param start an int, the starting dimension to flatten
 * @param stop an int, the end dimension to flatten
 * @return an fl::Variable with collasped dimensions
 */
Variable flatten(const fl::Variable& x, int start, int stop);

/**
 * Computes the generalizedBoxIou pairwise across to arrays
 * See: https://giou.stanford.edu/
 * @param: bboxes1 an array of shape [4, N, B]
 * @param: bboxes2 an array of shape [4, M, B]
 * @return an Tensor of shape [N x M x B] where each entry represents
 * the giou between two boxes
 */
Tensor generalizedBoxIou(const Tensor& bboxes1, const Tensor& bboxes2);

/**
 * Computes the generalizedBoxIou pairwise across to fl::Variables
 * See: https://giou.stanford.edu/
 * @param: bboxes1 an fl::Variable of shape [4, N, B]
 * @param: bboxes2 an fl::Variable of shape [4, M, B]
 * @return an fl::Variable of shape [N x M x B] where each entry represents
 * the giou between two boxes
 */
Variable generalizedBoxIou(const Variable& bboxes1, const Variable& bboxes2);

/**
 * Computes the iou pairwise across two arrays of bboxes
 * @param: bboxes1 an array of shape [4, N, B]
 * @param: bboxes2 an array of shape [4, M, B]
 * @return an tuple of Tensor of shape [N x M x B] where each entry
 * represents the iou and intersection between two boxes
 */
std::tuple<Tensor, Tensor> boxIou(
    const Tensor& bboxes1,
    const Tensor& bboxes2);

/**
 * Computes the iou pairwise across to Variables of bboxes
 * @param: bboxes1 an Variable of shape [4, N, B]
 * @param: bboxes2 an Variable of shape [4, M, B]
 * @return an tuple of fl::Variable of shape [N x M x B] where each entry
 * represents the iou and intersection between two boxes
 */
std::tuple<fl::Variable, fl::Variable> boxIou(
    const fl::Variable& bboxes1,
    const fl::Variable& bboxes2);

/**
 * Computes the l1_loss pairwise across two arrays of boxes
 * @param: bboxes1 an Variable of shape [4, N, B]
 * @param: bboxes2 an Variable of shape [4, M, B]
 * @return an tuple of fl::Variable of shape [N x M x B] where each entry
 * represents the l1Loss between two boxes
 */
Variable l1Loss(const Variable& input, const Variable& target);

} // namespace vision
} // namespace pkg
} // namespace fl
