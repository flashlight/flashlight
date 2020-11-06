/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <arrayfire.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Types.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {

class Variable;

namespace detail {

class ConvBenchmarks;

af::array tileAs(const af::array& input, const af::dim4& rdims);

af::array sumAs(const af::array& input, const af::dim4& rdims);

bool areVariableTypesEqual(const Variable& a, const Variable& b);

template <typename... Args>
bool areVariableTypesEqual(
    const Variable& a,
    const Variable& b,
    const Args&... args) {
  return areVariableTypesEqual(a, b) && areVariableTypesEqual(a, args...) &&
      areVariableTypesEqual(b, args...);
}

/**
 * Performs type conversion based on the optim level. Operations that lack
 * sufficient precision are automatically upcast to f32 before computation.
 * These are typically operations that require accumulations or reductions.
 */
template <typename T>
T adjustInputType(const T& in, const char* funcname) {
  OptimLevel optimLevel = OptimMode::get().getOptimLevel();
  // Fastpath - DEFAULT mode never casts tensors
  if (optimLevel == OptimLevel::DEFAULT) {
    return in;
  }

  T res;
  auto& funcs = kOptimLevelTypeExclusionMappings.find(optimLevel)->second;
  // TODO: tiny, but this lookup incurs an extra alloc from char* to string
  if (funcs.find(std::string(funcname)) == funcs.end() &&
      optimLevel != OptimLevel::DEFAULT) {
    // Not in the excluded list - cast to f16
    res = in.as(af::dtype::f16);
  } else {
    // Upcast to f32 only if we have an f16 input - otherwise, leave as is
    if (in.type() == af::dtype::f16) {
      res = in.as(af::dtype::f32);
    } else {
      res = in;
    }
  }

  return res;
}

} // namespace detail

/**
 * Adjusts the input type to operators based on the optimization mode and the
 * operator name.
 */
#define FL_ADJUST_INPUT_TYPE(INPUT) detail::adjustInputType(INPUT, __func__)

/**
 * Checks if a variadic number of Variables have the same types.
 */
#define FL_VARIABLE_DTYPES_MATCH_CHECK(...)              \
  if (!detail::areVariableTypesEqual(__VA_ARGS__)) {     \
    throw std::invalid_argument(                         \
        std::string(__func__) +                          \
        " doesn't support binary "                       \
        "operations with Variables of different types"); \
  }

/**
 * \defgroup autograd_functions Autograd Functions
 * @{
 */

/**
 * Element-wise addition of two Variables.
 * \f[ out = var_1 + var_2 \f]
 */
Variable operator+(const Variable& lhs, const Variable& rhs);

/**
 * Adds a scalar to each element in the Variable.
 * \f[ out_i = value + var_i \f]
 */
Variable operator+(const double& lhs, const Variable& rhs);

/**
 * Adds a scalar to each element in the Variable.
 * \f[ out_i = var_i + value \f]
 */
Variable operator+(const Variable& lhs, const double& rhs);

/**
 * Element-wise multiplication of two Variables.
 * \f[ out = var_1 \times var_2 \f]
 */
Variable operator*(const Variable& lhs, const Variable& rhs);

/**
 * Multiplies each element in the Variable by a scalar.
 * \f[ out_i = value \times var_i \f]
 */
Variable operator*(const double& lhs, const Variable& rhs);

/**
 * Multiplies each element in the Variable by a scalar.
 * \f[ out_i = var_i \times value \f]
 */
Variable operator*(const Variable& lhs, const double& rhs);

/**
 * Element-wise subtraction of two Variables.
 * \f[ out = var_1 - var_2 \f]
 */
Variable operator-(const Variable& lhs, const Variable& rhs);

/**
 * Subtracts a scalar from each element in the Variable.
 * \f[ out_i = var_i - value \f]
 */
Variable operator-(const double& lhs, const Variable& rhs);

/**
 * Subtracts each element in the Variable from a scalar.
 * \f[ out_i = value - var_i \f]
 */
Variable operator-(const Variable& lhs, const double& rhs);

/**
 * Element-wise division of two Variables.
 * \f[ out = \frac{var_1}{var_2} \f]
 */
Variable operator/(const Variable& lhs, const Variable& rhs);

/**
 * Divides each element in the Variable by a scalar.
 * \f[ out_i = \frac{var_i}{value} \f]
 */
Variable operator/(const double& lhs, const Variable& rhs);

/**
 * Divides a scalar by each element in the Variable.
 * \f[ out_i = \frac{value}{var_i} \f]
 */
Variable operator/(const Variable& lhs, const double& rhs);

/**
 * [Non-differentiable] Element-wise comparison of two Variables.
 * \f[ out = var_1 > var_2 \f]
 */
Variable operator>(const Variable& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = value > var_i \f]
 */
Variable operator>(const double& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = var_i > value \f]
 */
Variable operator>(const Variable& lhs, const double& rhs);

/**
 * [Non-differentiable] Element-wise comparison of two Variables.
 * \f[ out = var_1 < var_2 \f]
 */
Variable operator<(const Variable& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = value < var_i \f]
 */
Variable operator<(const double& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = var_i < value \f]
 */
Variable operator<(const Variable& lhs, const double& rhs);

/**
 * [Non-differentiable] Element-wise comparison of two Variables.
 * \f[ out = var_1 >= var_2 \f]
 */
Variable operator>=(const Variable& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = value >= var_i \f]
 */
Variable operator>=(const double& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = var_i >= value \f]
 */
Variable operator>=(const Variable& lhs, const double& rhs);

/**
 * [Non-differentiable] Element-wise comparison of two Variables.
 * \f[ out = var_1 <= var_2 \f]
 */
Variable operator<=(const Variable& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = value <= var_i \f]
 */
Variable operator<=(const double& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise comparison of a Variable and a scalar.
 * \f[ out_i = value <= var_i \f]
 */
Variable operator<=(const Variable& lhs, const double& rhs);

/**
 * [Non-differentiable] Element-wise logical and of two Variables.
 * \f[ out = var_1 \& var_2 \f]
 */
Variable operator&&(const Variable& lhs, const Variable& rhs);

/**
 * [Non-differentiable] Element-wise logical not of a Variable.
 * \f[ out_i = !var_i \f]
 */
Variable operator!(const Variable& input);

/**
 * Computes negative of each element in a Variable.
 * \f[ out_i = -var_i \f]
 */
Variable negate(const Variable& input);

/**
 * Computes reciprocal of each element in a Variable.
 * \f[ out_i = \frac{1}{var_i} \f]
 */
Variable reciprocal(const Variable& input);

/**
 * Computes exponential of each element in a Variable.
 * \f[ out_i = e^{var_i} \f]
 */
Variable exp(const Variable& input);

/**
 * Computes natural logarithm of each element in a Variable.
 * \f[ out_i = log(var_i) \f]
 */
Variable log(const Variable& input);

/**
 * Computes power of each element in a Variable.
 * \f[ out_i = var_i^p \f]
 */
Variable pow(const Variable& input, double p);

/**
 * Computes natural logarithm of (1 + element) for each element in a Variable.
 * \f[ out_i = log(1.0 + var_i) \f]
 */
Variable log1p(const Variable& input);

/**
 * Computes sine of each element in a Variable.
 * \f[ out_i = sin(var_i) \f]
 */
Variable sin(const Variable& input);

/**
 * Computes cosine of each element in a Variable.
 * \f[ out_i = cos(var_i) \f]
 */
Variable cos(const Variable& input);

/**
 * Computes square root of each element in a Variable.
 * \f[ out_i = \sqrt{var_i} \f]
 */
Variable sqrt(const Variable& input);

/**
 * Computes hyperbolic tangent of each element in a Variable.
 * \f[ out_i = \frac{\exp(var_i) - \exp(-var_i)}{\exp(var_i) + \exp(-var_i)} \f]
 */
Variable tanh(const Variable& input);

/**
 * Clamps all elements in input into the range [ `min`, `max` ] and return a
 * resulting tensor:
 * \f[ \begin{split}y_i = \begin{cases}
 *     \text{min} & \text{if } x_i < \text{min} \\
 *     x_i & \text{if } \text{min} \leq x_i \leq \text{max} \\
 *     \text{max} & \text{if } x_i > \text{max}
 *     \end{cases}\end{split} \f]
 */
Variable clamp(const Variable& input, const double min, const double max);

/**
 * Computes sigmoid of each element in a Variable.
 * \f[ out_i = \frac{1}{1 + \exp(-var_i)} \f]
 */
Variable sigmoid(const Variable& input);

/**
 * Returns element-wise maximum value of two Variables.
 * \f[ out = max(var_1, var_2) \f]
 */
Variable max(const Variable& lhs, const Variable& rhs);

/**
 * Returns maximum value of a scalar and each element in a Variable.
 * \f[ out_i = max(var_i, value) \f]
 */
Variable max(const Variable& lhs, const double& rhs);

/**
 * Returns maximum value of a scalar and each element in a Variable.
 * \f[ out_i = max(value, var_i) \f]
 */
Variable max(const double& lhs, const Variable& rhs);

/**
 * Returns element-wise minimum value of two Variables.
 * \f[ out = min(var_1, var_2) \f]
 */
Variable min(const Variable& lhs, const Variable& rhs);

/**
 * Returns minimum value of a scalar and each element in a Variable.
 * \f[ out_i = min(var_i, value) \f]
 */
Variable min(const Variable& lhs, const double& rhs);

/**
 * Returns minimum value of a scalar and each element in a Variable.
 * \f[ out_i = min(value, var_i) \f]
 */
Variable min(const double& lhs, const Variable& rhs);

/**
 * Returns a tensor that is a transposed version of a Variable. The first two
 * dimensions are swapped.
 */
Variable transpose(const Variable& input);

/**
 * Repeats the tensor `input` along certain dimensions so as to match the shape
 * of `reference`. The dimensions to be repeated along are automatically
 * inferred.
 */
Variable tileAs(const Variable& input, const Variable& reference);

/**
 * Repeats the tensor `input` along certain dimensions so as to match the shape
 * in the descriptor `rdims`. The dimensions to be repeated along are
 * automatically inferred.
 */
Variable tileAs(const Variable& input, const af::dim4& rdims);

/**
 * Sums up the tensor `input` along certain dimensions so as to match the shape
 * of `reference`. The dimensions to be summed along are automatically inferred.
 * Note that after summation, the shape of those dimensions will be 1.
 */
Variable sumAs(const Variable& input, const Variable& reference);

/**
 * Sums up the tensor `input` along certain dimensions so as to match the shape
 * in the descriptor `rdims`. The dimensions to be summed along are
 * automatically inferred. Note that after summation, the shape of those
 * dimensions will be 1.
 */
Variable sumAs(const Variable& input, const af::dim4& rdims);

/**
 * Concatenates Variables along a specific dimension. The shape of input
 * Variables should be identical except the dimension to concatenate.
 */
Variable concatenate(const std::vector<Variable>& concatInputs, int dim);

/**
 * Splits a Variable into equally sized chunks (if possible)
 *
 * @param input a Variable to split
 * @param splitSize size of each split. If input dimension is not evenly
 * divisible, last chunk of smaller splitSize will be included.
 * @param dim dimension along which to split the Variable
 */
std::vector<Variable> split(const Variable& input, dim_t splitSize, int dim);

/**
 * Splits a Variable into smaller chunks.
 *
 * @param input a Variable to split
 * @param splitSizes vector of integers specifying the sizes for each split
 * @param dim dimension along which to split the Variable
 */
std::vector<Variable>
split(const Variable& input, const std::vector<dim_t>& splitSizes, int dim);

/**
 * Repeats the tensor `input` along specific dimensions. The number of
 * repetition along each dimension is specified in descriptor `dims`.
 */
Variable tile(const Variable& input, const af::dim4& dims);

/**
 * Repeats the tensor `input` along specific dimensions. The number of
 * repetition along each dimension is specified in descriptor `dims`.
 *
 * @param[in] precision Type of the output vector when is it is desired to
 * be different from the input type. This is particularly useful when tile is
 * applied on parameters and the results will be used in a half precision
 * arithmetic.
 */
Variable
tile(const Variable& input, const af::dim4& dims, const af::dtype precision);

/**
 * Sums up the tensors `input` along dimensions specified in descriptor `axes`.
 * If `axes` has size greater than 1, reduce over all of them.
 */
Variable sum(const Variable& input, const std::vector<int>& axes);

/**
 * Computes the mean of the tensor `input` along dimensions specified in
 * descriptor `axes`. If `axes` has size greater than 1, reduce over all of
 * them.
 */
Variable mean(const Variable& input, const std::vector<int>& axes);

/**
 * Lp-norm computation, reduced over specified dimensions.
 * @param input tensor on which the Lp norm is going to be computed.
 * @param p the p value of the Lp norm.
 * @param axes dimensions over which the reduction is performed.
 */
Variable
norm(const Variable& input, const std::vector<int>& axes, double p = 2);

/**
 * Lp norm normalization, reduced over specified dimensions.
 * @param input the tensor to be normalized.
 * @param axes dimensions over which the reduction is performed.
 * @param p the p value of the Lp norm.
 * @param eps clamping value to avoid overflows.
 */
Variable normalize(
    const Variable& input,
    const std::vector<int>& axes,
    double p = 2,
    double eps = 1e-12);

/**
 * Computes variance of the tensor `input` along dimensions specified in
 * descriptor `axes`. If `axes` has size greater than 1, reduce over all of
 * them. Uses population variance if `isbiased` is `true`, otherwise, uses
 * sample variance.
 *
 * NB: the behavior of `fl::var` differs from that of `af::var`. In ArrayFire
 * versions >= 3.7.0, if `isbiased` is `true` the variance computation uses
 * sample variance; if `false`, population variance is used. For versions of
 * ArrayFire before v3.7.0, the reverse is true.
 */
Variable var(
    const Variable& input,
    const std::vector<int>& axes,
    const bool isbiased = false);

/**
 * Conducts matrix-matrix multiplication on two Variables. This is a batched
 * function if \f$B_1\f$ or \f$B_2\f$ is greater than 1.

 * @param lhs a Variable with shape [\f$M\f$, \f$N\f$, \f$B_1\f$, \f$B_2\f$]
 * @param rhs a Variable with shape [\f$N\f$, \f$K\f$, \f$B_1\f$, \f$B_2\f$]
 * @return a Variable with shape [\f$M\f$, \f$K\f$, \f$B_1\f$, \f$B_2\f$]
 */
Variable matmul(const Variable& lhs, const Variable& rhs);

/**
 * Conducts matrix-matrix multiplication on two Variables, where the first one
 * will be transposed before multiplication. This is a batched function if
 * \f$B_1\f$ or \f$B_2\f$ is greater than 1.

 * @param lhs a Variable with shape [\f$N\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 * @param rhs a Variable with shape [\f$N\f$, \f$K\f$, \f$B_1\f$, \f$B_2\f$]
 * @return a Variable with shape [\f$M\f$, \f$K\f$, \f$B_1\f$, \f$B_2\f$]
 */
Variable matmulTN(const Variable& lhs, const Variable& rhs);

/**
 * Conducts matrix-matrix multiplication on two Variables, where the second one
 * will be transposed before multiplication. This is a batched function if
 * \f$B_1\f$ or \f$B_2\f$ is greater than 1.

 * @param lhs a Variable with shape [\f$M\f$, \f$N\f$, \f$B_1\f$, \f$B_2\f$]
 * @param rhs a Variable with shape [\f$K\f$, \f$N\f$, \f$B_1\f$, \f$B_2\f$]
 * @return a Variable with shape [\f$M\f$, \f$K\f$, \f$B_1\f$, \f$B_2\f$]
 */
Variable matmulNT(const Variable& lhs, const Variable& rhs);

/**
 * Returns the absolute values of each element in a Variable.
 * \f[ out_i = |var_i| \f]
 */
Variable abs(const Variable& input);

/**
 * Flattens the input to a single dimension.
 */
Variable flat(const Variable& input);

/**
 * Modifies the input dimensions without changing the data order. The shape of
 * the output Variable is specified in descriptor `dims`.
 */
Variable moddims(const Variable& input, const af::dim4& dims);

/**
 * Exchanges data of an array such that the requested change in dimension is
 * satisfied. The linear ordering of data within the array is preserved.
 */
Variable reorder(
    const Variable& input,
    const int dim0,
    const int dim1,
    const int dim2 = 2,
    const int dim3 = 3);

/**
 * Applies a linear transformation to the input Variable:
 * \f[
 *    y = Ax
 * \f]
 * @param input a Variable with shape [\f$N\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 * @param weight a Variable with shape [\f$K\f$, \f$N\f$]
 * @return a Variable with shape [\f$K\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 */
Variable linear(const Variable& input, const Variable& weight);

/**
 * Applies a linear transformation to the input Variable:
 * \f[
 *    y = Ax + b
 * \f]
 * @param input a Variable with shape [\f$N\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 * @param weight a Variable with shape [\f$K\f$, \f$N\f$]
 * @param bias a Variable with shape [\f$K\f$]
 * @return a Variable with shape [\f$K\f$, \f$M\f$, \f$B_1\f$, \f$B_2\f$]
 */
Variable
linear(const Variable& input, const Variable& weight, const Variable& bias);

/**
 * Applies a 2D convolution over an input signal given filter weights. In the
 * simplest case, the output with shape [\f$X_{out}\f$, \f$Y_{out}\f$,
 * \f$C_{out}\f$, \f$N\f$] of the convolution with input [\f$X_{in}\f$,
 * \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$, \f$K_y\f$,
 * \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]

 * @param input a Variable with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$,
 * \f$N\f$]
 * @param weights a Variable with shape [\f$K_x\f$, \f$K_y\f$, \f$C_{in}\f$,
 * \f$C_{out}\f$]
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param dx dilation along the first kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param dy dilation along the second kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param groups number of filter groups
 * @param benchmarks [optional] a `ConvBenchmarks` instance to use to
 * dynamically benchmark configuration attributes for computations.
 * @return a Variable with shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C_{out}\f$,
 * \f$N\f$]]
 */
Variable conv2d(
    const Variable& input,
    const Variable& weights,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    int dx = 1,
    int dy = 1,
    int groups = 1,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks = nullptr);

/**
 * Applies a 2D convolution over an input signal given filter weights and
 * biases. In the simplest case, the output with shape [\f$X_{out}\f$,
 * \f$Y_{out}\f$, \f$C_{out}\f$, \f$N\f$] of the convolution with input
 * [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$, \f$N\f$] and weight [\f$K_x\f$,
 * \f$K_y\f$, \f$C_{in}\f$, \f$C_{out}\f$] can be precisely described as:
 * \f[
      \text{out}(C_{out_j}, N_i) =
          \text{bias}(C_{out_j}) +
          \sum_{k = 0}^{C_{in} - 1} \text{weight}(k, C_{out_j}) \star
          \text{input}(k, N_i)
 * \f]

 * @param input a Variable with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C_{in}\f$,
 * \f$N\f$]
 * @param weights a Variable with shape [\f$K_x\f$, \f$K_y\f$, \f$C_{in}\f$,
 * \f$C_{out}\f$]
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param dx dilation along the first kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param dy dilation along the second kernel dimension. A dilation of 1
 * is equivalent to a standard convolution along this axis.
 * @param groups number of filter groups
 * @param benchmarks [optional] a `ConvBenchmarks` instance to use to
 * dynamically benchmark configuration attributes for computations.
 * @param bias a Variable with shape [\f$C_{out}\f$]
 * @return a Variable with shape [\f$X_{out}\f$, \f$Y_{out}\f$, \f$C_{out}\f$,
 * \f$N\f$]]
 */
Variable conv2d(
    const Variable& input,
    const Variable& weights,
    const Variable& bias,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    int dx = 1,
    int dy = 1,
    int groups = 1,
    std::shared_ptr<detail::ConvBenchmarks> benchmarks = nullptr);

/**
 * Applies a 2D pooling over an input signal composed of several input planes.
 * @param input a Variable with shape [\f$X_{in}\f$, \f$Y_{in}\f$, \f$C\f$,
 *   \f$N\f$]
 * @param wx pooling window size in the first dimension
 * @param wy pooling window size in the second dimension
 * @param sx stride in the first dimension
 * @param sy stride in the second dimension
 * @param px number of positions of zero-padding on both sides in the first
 * dimension
 * @param py number of positions of zero-padding on both sides in the second
 * dimension
 * @param mode pooling mode, which supports:
 * - MAX
 * - AVG_INCLUDE_PADDING
 * - AVG_EXCLUDE_PADDING
 */
Variable pool2d(
    const Variable& input,
    int wx,
    int wy,
    int sx = 1,
    int sy = 1,
    int px = 0,
    int py = 0,
    PoolingMode mode = PoolingMode::MAX);

/**
 * Applies a softmax function on Variable `input` along dimension `dim`, so that
 * the elements of the dimensional `dim` in output lie in the range (0,1) and
 * sum to 1.
 * \f[ out(x_{i}) = \frac{exp(x_i)}{\sum_j exp(x_j)} \f]
 */
Variable softmax(const Variable& input, const int dim);

/**
 * Applies a log(softmax(x)) function on Variable `input` along dimension `dim`
 * \f[
 *    out(x_{i}) = log \Big( \frac{exp(x_i)}{\sum_j exp(x_j)} \Big)
 * \f]
 */
Variable logSoftmax(const Variable& input, const int dim);

/**
 * Computes the binary cross entropy loss between an input tensor \f$x\f$ and a
 * target tensor \f$y\f$. The binary cross entropy loss is:
 * \f[
      B(x, y) = \frac{1}{n} \sum_{i = 0}^n
      -\left( y_i \times \log(x_i) + (1 - y_i) \times \log(1 - x_i) \right)
   \f]
 * Both the inputs and the targets are expected to be between 0 and 1.
 *
 * @param inputs a tensor with the predicted values
 * @param targets a tensor with the target values
 */
Variable binaryCrossEntropy(const Variable& inputs, const Variable& targets);

/**
 * Computes the categorical cross entropy loss. The input is expected to
 * contain log-probabilities for each class. The targets should be the
 * index of the ground truth class for each input example.
 * \f[
 *   \begin{split}\ell(x, y) = \begin{cases}
 *      \frac{1}{N} \sum_{n=1}^N -x_{n,y_n}, & \text{if}\;
 *      \text{reduction} = \text{MEAN},\\
 *      \sum_{n=1}^N -x_{n,y_n},  & \text{if}\;
 *      \text{reduction} = \text{SUM}, \\
 *      \{ -x_{1,y_1}, ..., -x_{N,y_N} \},  & \text{if}\;
 *      \text{reduction} = \text{NONE}.
 *   \end{cases}\end{split}
 * \f]

 * @param input a `Variable` with shape [\f$C\f$, \f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$] where \f$C\f$ is the number of classes.
 * @param targets an integer `Variable` with shape [\f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$]. The values must be in \f$[0, C - 1]\f$
 * @param reduction reduction mode, which supports:
 * - NONE
 * - MEAN
 * - SUM
 * @param ignoreIndex a target value that is ignored and does not contribute
 * to the loss or the input gradient. If `reduce` is MEAN, the loss is
 * averaged over non-ignored targets. Only indicies in \f$[0, C - 1]\f$ are
 * considered to be valid.
 * @return a `Variable` of loss value with shape scalar by default. If `reduce`
 * is NONE, then [\f$B_1\f$, \f$B_2\f$, \f$B_3\f$].
 */
Variable categoricalCrossEntropy(
    const Variable& input,
    const Variable& targets,
    ReduceMode reduction = ReduceMode::MEAN,
    int ignoreIndex = -1);

/**
 * The gated linear unit.
 * \f[
        H = A \times \sigma(B)
 * \f]
 * where `input` is split in half along `dim` to form `A` and `B`.
 * See [Language Modeling with Gated Convolutional Networks]
 * (https://arxiv.org/abs/1612.08083).

 * @param input input Variable
 * @param dim dimension on which to split the input
 */
Variable gatedlinearunit(const Variable& input, const int dim);

/**
 * Applies an RNN unit to an input sequence.
 * A general RNN operator can be expressed as following:
 * \f[
      (h_t, c_t) = f_W(x_t, h_{t-1}, c_{t-1})
 * \f]
 * where \f$h_t\f$, \f$c_t\f$ are the hidden/cell state at time \f$t\f$,
 * \f$x_t\f$ is the input at time \f$t\f$

 * @param input Variable of input with shape [input size, batch size, sequence
 * length]
 * @param hiddenState Variable of hidden state with shape [hidden size, batch
 * size, total layers]
 * @param cellState [LSTM only] Variable of cell state with same shape as
 * hidden state
 * @param weights Learnable parameters in the RNN unit
 * @param hiddenSize number of features in the hidden state
 * @param numLayers number of recurrent layers
 * @param mode defines the type of RNN unit
 *  - RELU
 *  - TANH
 *  - LSTM
 *  - GRU
 * @param bidirectional if `True`, becomes a bidirectional RNN, unidirectional
 otherwise
 * @param dropout if non-zero, introduces a `Dropout` layer on the outputs of
 * each RNN layer except the last one, with dropout probability equal to dropout

 * @return a tuple of three Variables:
 * - `y`: input with shape [input size, batch size, sequence length *
 * directions]
 * - `hiddenState`: hidden state for the current time step
 * - `cellState`: cell state for the current time step
 */
std::tuple<Variable, Variable, Variable> rnn(
    const Variable& input,
    const Variable& hiddenState,
    const Variable& cellState,
    const Variable& weights,
    int hiddenSize,
    int numLayers,
    RnnMode mode,
    bool bidirectional,
    float dropout);

/**
 * Looks up embeddings in a fixed dictionary and size.
 * @param input a Variable of a list of indices with shape [\f$B_1\f$,
 * \f$B_2\f$, \f$B_3\f$]
 * @param embeddings a Variable of an embedding matrix with shape [\f$D\f$,
 * \f$N\f$], where \f$N\f$ is the number of items and \f$D\f$ is the embedding
 * size.
 * @return a Variable of embeddings with shape [\f$D\f$, \f$B_1\f$, \f$B_2\f$,
 * \f$B_3\f$]
 */
Variable embedding(const Variable& input, const Variable& embeddings);

/**
 * Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with
 * additional channel dimension) as described in the paper
 * [Batch Normalization: Accelerating Deep Network Training by Reducing Internal
 * Covariate Shift] (https://arxiv.org/abs/1502.03167) .
 * \f[
 *   y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma +
 * \beta
 * \f]
 * The mean and standard-deviation are calculated per-dimension over the
 * mini-batches and \f$\gamma\f$ and \f$\beta\f$ are learnable parameter vectors
 * of size \f$C\f$, the input size. By default, during training this layer keeps
 * running estimates of its computed mean and variance, which are then used for
 * normalization during evaluation.

 * @param input a Variable with size [\f$H\f$, \f$W\f$, \f$C\f$, \f$N\f$]
 * @param weight a Variable with size [\f$C\f$] for \f$\gamma\f$
 * @param bias a Variable with size [\f$C\f$] for \f$\beta\f$
 * @param runningMean a buffer storing intermediate means during training
 * @param runningVar a buffer storing intermediate variances during training
 * @param axes dimensions to perform normalization on. If having size greater
 * than one, reduce over all of them.
 * @param train a flag indicating if running in training mode
 * @param momentum value of momentum
 * @param epsilon value of \f$\epsilon\f$

 * @return a Variable with same shape as `input`
 */
Variable batchnorm(
    const Variable& input,
    const Variable& weight,
    const Variable& bias,
    Variable& runningMean,
    Variable& runningVar,
    const std::vector<int>& axes,
    bool train,
    double momentum,
    double epsilon);

/**
 * Applies asymmetric padding on a Variable `input`.
 * @param input input Variable
 * @param pad a list of integer pairs specifying the positions we want to pad on
 * both sides for each dimension
 * @param val padding value
 * @return a padded Variable
 */
Variable padding(
    const Variable& input,
    std::vector<std::pair<int, int>> pad,
    double val);

/**
 * Applies dropout on a Variable `input`.
 * @param input input Variable
 * @param p the probability of dropout
 * @return a droped out Variable
 */
Variable dropout(const Variable& input, double p);

/**
 * Applies the [rectified linear
 * unit](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function
 * element-wise to a `Variable`:
 * \f[ ReLU(x) = \max(0, x) \f]
 */
Variable relu(const Variable& input);

/**
 * Applies the [Gaussian Error linear
 * Unit](https://arxiv.org/abs/1606.08415) function
 * element-wise to a `Variable`
 */
Variable gelu(const Variable& input);

/**
 * Relative positional embedding for the multihead attention
 * Implementation partially follows https://arxiv.org/pdf/1803.02155.pdf
 */
Variable relativePositionalEmbeddingRotate(const Variable& input);

/**
 * Multihead Attention function
 * For details, see [Vaswani et al (2017)](https://arxiv.org/abs/1706.03762).
 * @param query query Variable
 * @param key key Variable
 * @param value value Variable
 * @param posEmb if non empty then compute relative
 * positional embedding in additon to standard computations
 * @param mask mask or not future in the computations
 * if non-empty then don't use future (for exmaple for autoregressive language models
 * or for decoder part in the encoder-decoder transformer models)
 * @param nHeads number of heads
 * @param pDropout dropout probability
 * @param offset size of the current output from the decoder used now as input
 */
Variable multiheadAttention(
    const Variable& query,
    const Variable& key,
    const Variable& value,
    const Variable& posEmb,
    const Variable& mask,
    const int32_t nHeads,
    const double pDropout,
    const int32_t offset = 0);

/** @} */

} // namespace fl
