/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/tensor/backend/jit/ShapeInference.h"

#include <cassert>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace fl {

Shape inferReductionOutputShape(
    const Shape& inputShape,
    const std::vector<int>& axes,
    bool keepDims) {
  // NOTE the tensor API requires this, see reduction tests
  if (inputShape.ndim() == 0) {
    return inputShape;
  }
  for (const auto axis : axes) {
    if (axis < 0 || axis >= inputShape.ndim()) {
      std::ostringstream oss;
      oss << "[inferReductionOutputShape] Invalid axis for reduction: " << axis
          << " for tensor of shape: " << inputShape;
      throw std::invalid_argument(oss.str());
    }
  }
  std::unordered_set<int> axesToReduce;
  if (axes.empty()) {
    for (int aixs = 0; aixs < inputShape.ndim(); aixs++) {
      axesToReduce.insert(aixs);
    }
  } else {
    axesToReduce.insert(axes.begin(), axes.end());
  }
  std::vector<Dim> outputDims;
  for (int axis = 0; axis < inputShape.ndim(); axis++) {
    if (axesToReduce.find(axis) != axesToReduce.end()) {
      if (keepDims) {
        outputDims.push_back(1);
      }
    } else {
      outputDims.push_back(inputShape.dim(axis));
    }
  }
  return Shape(outputDims);
}

std::vector<Shape> broadcastShapesToMaxRank(
    const std::vector<Tensor>& tensors,
    const unsigned minOutputRank) {
  assert(!tensors.empty());
  std::vector<Shape> shapes;
  int maxRank = minOutputRank;
  for (const auto& tensor : tensors) {
    if (tensor.ndim() == 0) {
      // TODO investigate semantics for concatnating empty tensor
      throw std::runtime_error(
          "[broadcastShapesToMaxRank] Currently doesn't support empty tensor");
    }
    maxRank = std::max(maxRank, tensor.ndim());
  }
  for (const auto& tensor : tensors) {
    auto dims = tensor.shape().get();
    dims.insert(dims.end(), maxRank - dims.size(), 1);
    shapes.push_back(Shape(dims));
  }
  return shapes;
}

Shape inferTransposeOutputShape(const Shape& inputShape, const Shape& axes) {
  const auto inputRank = inputShape.ndim();
  std::vector<Dim> outputDims = inputShape.get();
  std::vector<Dim> oldToNewAxes = axes.get();
  if (axes.ndim() == 0) { // default, reverse all axes
    oldToNewAxes.resize(inputRank);
    std::reverse(outputDims.begin(), outputDims.end());
    std::iota(oldToNewAxes.begin(), oldToNewAxes.end(), 0);
    std::reverse(oldToNewAxes.begin(), oldToNewAxes.end());
  } else if (axes.ndim() == inputRank) {
    for (int axis = 0; axis < axes.ndim(); axis++) {
      outputDims[axis] = inputShape.dim(oldToNewAxes[axis]);
    }
  } else {
    std::ostringstream oss;
    oss << "[JitBackend::transpose] Invalid axes: " << axes
        << " for shape: " << inputShape;
    throw std::runtime_error(oss.str());
  }
  return Shape(outputDims);
}

Shape inferTileOutputShape(const Shape& inputShape, const Shape& tileDims) {
  std::vector<Dim> paddedTensorDims = inputShape.get();
  std::vector<Dim> paddedTileDims = tileDims.get();
  const auto inputRank = inputShape.ndim();
  const auto tileRank = tileDims.ndim();
  if (inputRank > tileRank) {
    const auto diff = inputRank - tileRank;
    paddedTileDims.insert(paddedTileDims.end(), diff, 1);
  } else {
    const auto diff = tileRank - inputRank;
    paddedTensorDims.insert(paddedTensorDims.end(), diff, 1);
  }
  std::vector<Dim> outputDims;
  for (unsigned i = 0; i < paddedTensorDims.size(); i++) {
    outputDims.push_back(paddedTensorDims[i] * paddedTileDims[i]);
  }
  return Shape(outputDims);
}

Shape inferConcatenateOutputShape(
    const std::vector<Tensor>& tensors,
    const unsigned axisToConcat) {
  // TODO need a nice way to construct empty tensor for wrapped backend
  if (tensors.empty()) {
    throw std::runtime_error(
        "[inferConcatenateOutputShape] Nothing to concatenate");
  }
  const auto shapes = broadcastShapesToMaxRank(tensors, axisToConcat + 1);
  const unsigned outputRank = shapes.front().ndim();
  if (axisToConcat >= outputRank) {
    std::ostringstream oss;
    oss << "[inferConcatenateOutputShape] Axis too big: " << axisToConcat;
    throw std::runtime_error(oss.str());
  }
  std::vector<Dim> outputDims;
  for (unsigned axis = 0; axis < outputRank; axis++) {
    Dim concatenatedDim = 0;
    if (axis == axisToConcat) {
      for (const auto& shape : shapes) {
        concatenatedDim += shape.dim(axis);
      }
    } else {
      concatenatedDim = shapes.front().dim(axis);
    }
    outputDims.push_back(concatenatedDim);
  }
  return Shape(outputDims);
}

Shape inferPadOutputShape(
    const Shape& inputShape,
    const std::vector<std::pair<int, int>>& padWidths) {
  std::vector<Dim> outputDims = inputShape.get();
  if (padWidths.size() > static_cast<size_t>(inputShape.ndim())) {
    throw std::runtime_error("[inferPadOutputShape] too many paddings");
  }
  for (unsigned axis = 0; axis < padWidths.size(); axis++) {
    const auto& [beforeDim, afterDim] = padWidths[axis];
    outputDims[axis] += beforeDim + afterDim;
  }
  return Shape(outputDims);
}

Shape inferMatmulOutputShape(
    const Shape& lhsShape,
    const Shape& rhsShape,
    MatrixProperty lhsProp,
    MatrixProperty rhsProp) {
  std::vector<Dim> lhsDims = lhsShape.get();
  std::vector<Dim> rhsDims = rhsShape.get();
  const auto lhsRank = lhsDims.size();
  const auto rhsRank = rhsDims.size();
  const bool isLhsScalarOrVector = lhsRank <= 1;
  const bool isRhsScalarOrVector = rhsRank <= 1;
  if (isLhsScalarOrVector) { // pad to (1 x 1/K)
    lhsDims.insert(lhsDims.end(), 2 - lhsRank, 1);
    std::reverse(lhsDims.begin(), lhsDims.end());
  } else if (lhsProp == MatrixProperty::Transpose) {
    std::swap(lhsDims[0], lhsDims[1]);
  }
  if (isRhsScalarOrVector) { // pad to (1/K x 1)
    rhsDims.insert(rhsDims.end(), 2 - rhsRank, 1);
  } else if (rhsProp == MatrixProperty::Transpose) {
    std::swap(rhsDims[0], rhsDims[1]);
  }
  // shape compatibility check
  bool onlyOneIsBatched = lhsRank >= 3 ^ rhsRank >= 3;
  if (!(lhsDims.at(1) == rhsDims.at(0) &&
        (onlyOneIsBatched ||
         std::equal(
             lhsDims.begin() + 2,
             lhsDims.end(),
             rhsDims.begin() + 2,
             rhsDims.end())))) {
    std::ostringstream oss;
    oss << "Cannot perform matmul for tensors of shapes: " << lhsShape
        << " and " << rhsShape;
    throw std::invalid_argument(oss.str());
  }
  // take dims from bigger rank in case batch broadcasting occurred
  std::vector<Dim> outputDims;
  if (lhsRank > rhsRank) {
    outputDims = lhsDims;
    outputDims[1] = rhsDims[1];
  } else {
    outputDims = rhsDims;
    outputDims[0] = lhsDims[0];
  }
  Shape outputShape(outputDims);
  if (isLhsScalarOrVector || isRhsScalarOrVector) {
    outputShape = {outputShape.elements()};
  }
  return outputShape;
}

} // namespace fl
