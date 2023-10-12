/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"

// TODO refactor all this with logic in other backends (AF & OneDNN), and move
// to a `common` folder all backend implementations can access and reuse.
namespace fl {

Shape inferReductionOutputShape(
    const Shape& inputShape,
    const std::vector<int>& axes,
    bool keepDims);

std::vector<Shape> broadcastShapesToMaxRank(
    const std::vector<Tensor>& tensors,
    const unsigned minOutputRank);

Shape inferTransposeOutputShape(const Shape& inputShape, const Shape& axes);

Shape inferTileOutputShape(const Shape& inputShape, const Shape& tileDims);

Shape inferConcatenateOutputShape(
    const std::vector<Tensor>& tensors,
    const unsigned axisToConcat);

Shape inferPadOutputShape(
    const Shape& inputShape,
    const std::vector<std::pair<int, int>>& padWidths);

Shape inferMatmulOutputShape(
    const Shape& lhsShape,
    const Shape& rhsShape,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp);

} // namespace fl
