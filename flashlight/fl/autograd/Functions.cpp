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

#include <algorithm>
#include <array>
#include <stdexcept>

#include <af/internal.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/autograd/Variable.h"

namespace fl {
namespace detail {

af::array tileAs(const af::array& input, const af::dim4& rdims) {
  af::dim4 dims(1, 1, 1, 1);
  af::dim4 idims = input.dims();
  for (int i = 0; i < 4; i++) {
    if (rdims[i] % idims[i] != 0) {
      std::stringstream ss;
      ss << "Invalid dims for tileAs for input dims " << idims
         << " to output dims " << rdims;
      throw std::invalid_argument(ss.str());
    }
    dims[i] = rdims[i] / idims[i];
  }
  return tile(input, dims);
}

af::array sumAs(const af::array& input, const af::dim4& rdims) {
  af::dim4 idims = input.dims();
  auto result = input;
  for (int i = 0; i < 4; i++) {
    if (idims[i] != rdims[i]) {
      if (rdims[i] != 1) {
        std::stringstream ss;
        ss << "Invalid dims for sumAs for input dims " << idims
           << " to output dims " << rdims;
        throw std::invalid_argument(ss.str());
      }
      result = af::sum(result, i);
    }
  }

  return result.as(input.type());
}

bool areVariableTypesEqual(const Variable& a, const Variable& b) {
  return a.type() == b.type();
}

} // namespace detail

Variable operator+(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() + rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(gradOutput.array(), false));
    inputs[1].addGrad(Variable(gradOutput.array(), false));
  };
  return Variable(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Variable operator+(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() + rhsVal).as(lhs.type());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(gradOutput.array(), false));
  };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator+(const double& lhsVal, const Variable& rhs) {
  return rhs + lhsVal;
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() - rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(gradOutput.array(), false));
    inputs[1].addGrad(Variable(negate(gradOutput).array(), false));
  };
  return Variable(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Variable operator-(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() - rhsVal).as(lhs.type());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(gradOutput.array(), false));
  };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator-(const double& lhsVal, const Variable& rhs) {
  auto result = (lhsVal - rhs.array()).as(rhs.type());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(negate(gradOutput).array(), false));
  };
  return Variable(result, {rhs.withoutData()}, gradFunc);
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() * rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    if (inputs[0].isCalcGrad()) {
      inputs[0].addGrad(Variable((gradOutput * inputs[1]).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      inputs[1].addGrad(Variable((gradOutput * inputs[0]).array(), false));
    }
  };
  return Variable(
      result,
      {rhs.isCalcGrad() ? lhs : lhs.withoutData(),
       lhs.isCalcGrad() ? rhs : rhs.withoutData()},
      gradFunc);
}

Variable operator*(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() * rhsVal).as(lhs.type());
  auto gradFunc =
      [rhsVal](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable((gradOutput * rhsVal).array(), false));
      };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator*(const double& lhsVal, const Variable& rhs) {
  return rhs * lhsVal;
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() / rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    auto inputs1rec = reciprocal(inputs[1]);
    auto gradInput0 = gradOutput * inputs1rec;
    if (inputs[0].isCalcGrad()) {
      inputs[0].addGrad(Variable(gradInput0.array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      inputs[1].addGrad(Variable(
          (gradInput0 * negate(inputs[0]) * inputs1rec).array(), false));
    }
  };
  return Variable(
      result, {rhs.isCalcGrad() ? lhs : lhs.withoutData(), rhs}, gradFunc);
}

Variable operator/(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() / rhsVal).as(lhs.type());
  auto gradFunc =
      [rhsVal](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable((gradOutput / rhsVal).array(), false));
      };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator/(const double& lhsVal, const Variable& rhs) {
  auto result = (lhsVal / rhs.array()).as(rhs.type());
  auto gradFunc =
      [lhsVal](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(
            (gradOutput * (-lhsVal) / (inputs[0] * inputs[0])).array(), false));
      };
  return Variable(result, {rhs}, gradFunc);
}

Variable operator>(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() > rhs.array();
  return Variable(result, false);
}

Variable operator>(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() > rhsVal).as(lhs.type());
  return Variable(result, false);
}

Variable operator>(const double& lhsVal, const Variable& rhs) {
  auto result = (lhsVal > rhs.array()).as(rhs.type());
  return Variable(result, false);
}

Variable operator<(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() < rhs.array();
  return Variable(result, false);
}

Variable operator<(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() < rhsVal).as(lhs.type());
  return Variable(result, false);
}

Variable operator<(const double& lhsVal, const Variable& rhs) {
  auto result = (lhsVal < rhs.array()).as(rhs.type());
  return Variable(result, false);
}

Variable operator>=(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() >= rhs.array();
  return Variable(result, false);
}

Variable operator>=(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() >= rhsVal).as(lhs.type());
  return Variable(result, false);
}

Variable operator>=(const double& lhsVal, const Variable& rhs) {
  auto result = (lhsVal >= rhs.array()).as(rhs.type());
  return Variable(result, false);
}

Variable operator<=(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() <= rhs.array();
  return Variable(result, false);
}

Variable operator<=(const Variable& lhs, const double& rhsVal) {
  auto result = (lhs.array() <= rhsVal).as(lhs.type());
  return Variable(result, false);
}

Variable operator<=(const double& lhsVal, const Variable& rhs) {
  auto result = (lhsVal <= rhs.array()).as(rhs.type());
  return Variable(result, false);
}

Variable operator&&(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = lhs.array() && rhs.array();
  return Variable(result, false);
}

Variable operator!(const Variable& input) {
  auto result = (!input.array()).as(input.type());
  return Variable(result, false);
}

Variable max(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = af::max(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    auto mask = Variable(
        (inputs[0].array() > inputs[1].array()).as(gradOutput.type()), false);
    inputs[0].addGrad(Variable((mask * gradOutput).array(), false));
    inputs[1].addGrad(Variable((!mask * gradOutput).array(), false));
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable max(const Variable& lhs, const double& rhsVal) {
  auto result = af::max(lhs.array(), rhsVal).as(lhs.type());
  auto gradFunc =
      [rhsVal](std::vector<Variable>& inputs, const Variable& gradOutput) {
        auto mask =
            Variable((inputs[0].array() > rhsVal).as(gradOutput.type()), false);
        inputs[0].addGrad(Variable((mask * gradOutput).array(), false));
      };
  return Variable(result, {lhs}, gradFunc);
}

Variable max(const double& lhsVal, const Variable& rhs) {
  return max(rhs, lhsVal);
}

Variable min(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  auto result = af::min(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    auto mask = Variable(
        (inputs[0].array() < inputs[1].array()).as(gradOutput.type()), false);
    inputs[0].addGrad(Variable((mask * gradOutput).array(), false));
    inputs[1].addGrad(Variable((!mask * gradOutput).array(), false));
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable min(const Variable& lhs, const double& rhsVal) {
  auto result = af::min(lhs.array(), rhsVal).as(lhs.type());
  auto gradFunc =
      [rhsVal](std::vector<Variable>& inputs, const Variable& gradOutput) {
        auto mask =
            Variable((inputs[0].array() < rhsVal).as(gradOutput.type()), false);
        inputs[0].addGrad(Variable((mask * gradOutput).array(), false));
      };
  return Variable(result, {lhs}, gradFunc);
}

Variable min(const double& lhsVal, const Variable& rhs) {
  return min(rhs, lhsVal);
}

Variable negate(const Variable& input) {
  auto result = (0.0 - input.array()).as(input.type());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(negate(gradOutput).array(), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable reciprocal(const Variable& input) {
  auto result = 1.0 / FL_ADJUST_INPUT_TYPE(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    auto res = reciprocal(inputs[0]);
    inputs[0].addGrad(
        Variable((negate(gradOutput) * res * res).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable exp(const Variable& input) {
  auto result = af::exp(FL_ADJUST_INPUT_TYPE(input.array()));
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(
        Variable(gradOutput.array() * af::exp(inputs[0].array()), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable log(const Variable& input) {
  auto result = af::log(FL_ADJUST_INPUT_TYPE(input.array()));
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable((gradOutput / inputs[0]).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable log1p(const Variable& input) {
  auto result = af::log1p(FL_ADJUST_INPUT_TYPE(input.array()));
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(
        Variable((gradOutput / (1.0 + inputs[0])).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable pow(const Variable& input, double p) {
  auto result = af::pow(FL_ADJUST_INPUT_TYPE(input.array()), p);
  auto gradFunc = [p](std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    af::array grad = p * af::pow(inputs[0].array(), p - 1) * gradOutput.array();
    inputs[0].addGrad(Variable(grad, false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable sin(const Variable& input) {
  auto result = af::sin(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable((gradOutput * cos(inputs[0])).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable cos(const Variable& input) {
  auto result = af::cos(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(
        Variable((gradOutput * negate(sin(inputs[0]))).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable tanh(const Variable& input) {
  auto result = af::tanh(input.array());
  auto gradFunc =
      [result](std::vector<Variable>& inputs, const Variable& gradOutput) {
        auto grad =
            Variable((1.0 - result * result) * gradOutput.array(), false);
        inputs[0].addGrad(Variable(grad.array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable clamp(const Variable& input, const double lo, const double hi) {
  auto result = af::clamp(input.array(), lo, hi);
  auto gradFunc = [lo, hi, result](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    af::array gradMask = gradOutput.array();
    replace(gradMask, (result > lo) && (result < hi), 0);
    inputs[0].addGrad(Variable(gradMask, false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sqrt(const Variable& input) {
  auto result = af::sqrt(input.array());
  auto gradFunc =
      [result](std::vector<Variable>& inputs, const Variable& gradOutput) {
        auto output = Variable(result, false);
        inputs[0].addGrad(Variable((gradOutput / (2 * output)).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sigmoid(const Variable& input) {
  auto result = af::sigmoid(input.array());
  auto gradFunc =
      [result](std::vector<Variable>& inputs, const Variable& gradOutput) {
        auto grad = gradOutput.array() * result * (1 - result);
        inputs[0].addGrad(Variable(grad, false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable transpose(const Variable& input) {
  auto result = af::transpose(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    inputs[0].addGrad(Variable(transpose(gradOutput).array(), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable tileAs(const Variable& input, const af::dim4& rdims) {
  auto result = detail::tileAs(input.array(), rdims);

  af::dim4 inDims = input.dims();
  auto gradFunc =
      [inDims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(
            sumAs(gradOutput, inDims).array().as(inputs[0].type()), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable tileAs(const Variable& input, const Variable& reference) {
  return tileAs(input, reference.dims());
}

Variable sumAs(const Variable& input, const af::dim4& rdims) {
  auto result = detail::sumAs(FL_ADJUST_INPUT_TYPE(input.array()), rdims);
  auto idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(tileAs(gradOutput, idims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sumAs(const Variable& input, const Variable& reference) {
  return sumAs(input, reference.dims());
}

Variable concatenate(const std::vector<Variable>& concatInputs, int dim) {
  if (concatInputs.empty()) {
    throw std::invalid_argument("cannot concatenate zero variables");
  }
  if (dim < 0 || dim > 3) {
    throw std::invalid_argument("invalid dimension to concatenate along");
  }
  if (concatInputs.size() == 1) {
    return concatInputs[0];
  }
  // All variables must be of the same type
  af::dtype type = concatInputs[0].type();
  for (auto& var : concatInputs) {
    if (var.type() != type) {
      throw std::invalid_argument(
          "concatenate: all input Variables must be of the same type");
    }
  }
  auto dims = concatInputs[0].dims();
  int concatSize = dims[dim];
  for (int i = 1; i < concatInputs.size(); i++) {
    concatSize += concatInputs[i].dims(dim);
    for (int d = 0; d < 4; d++) {
      if (dim != d && concatInputs[i].dims(d) != dims[d]) {
        throw std::invalid_argument(
            "mismatch in dimension not being concatenated");
      }
    }
  }
  dims[dim] = concatSize;
  af::array result(dims, concatInputs[0].type());
  std::array<af::index, 4> slice{af::span, af::span, af::span, af::span};
  int start = 0;
  for (const auto& input : concatInputs) {
    slice[dim] = af::seq(start, start + input.dims(dim) - 1);
    result(slice[0], slice[1], slice[2], slice[3]) = input.array();
    start += input.dims(dim);
  }

  std::vector<Variable> inputsNoData;
  std::vector<af::dim4> inDims;

  for (const auto& in : concatInputs) {
    inputsNoData.push_back(in.withoutData());
    inDims.push_back(in.dims());
  }

  auto gradFunc =
      [dim, inDims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        std::array<af::index, 4> sx{af::span, af::span, af::span, af::span};
        int s = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
          sx[dim] = af::seq(s, s + inDims[i][dim] - 1);
          inputs[i].addGrad(
              Variable(gradOutput.array()(sx[0], sx[1], sx[2], sx[3]), false));
          s += inDims[i][dim];
        }
      };

  return Variable(result, inputsNoData, gradFunc);
}

std::vector<Variable> split(const Variable& input, dim_t splitSize, int dim) {
  if (splitSize <= 0) {
    throw std::invalid_argument("split size must be a positive integer");
  }
  auto dimSize = input.dims(dim);
  std::vector<dim_t> splitSizes(dimSize / splitSize, splitSize);

  if (dimSize % splitSize > 0) {
    splitSizes.push_back(dimSize % splitSize);
  }
  return split(input, splitSizes, dim);
}

std::vector<Variable>
split(const Variable& input, const std::vector<dim_t>& splitSizes, int dim) {
  auto dimSize = input.dims(dim);
  auto N = splitSizes.size();

  std::vector<Variable> outputs(N);
  std::array<af::seq, 4> sel = {af::span, af::span, af::span, af::span};
  int start = 0;
  for (int i = 0; i < N; ++i) {
    if (splitSizes[i] <= 0) {
      throw std::invalid_argument("elements in split sizes has to be positive");
    }
    int end = start + splitSizes[i];
    sel[dim] = af::seq(start, end - 1);
    outputs[i] = input(sel[0], sel[1], sel[2], sel[3]);
    start = end;
  }
  if (start != dimSize) {
    throw std::invalid_argument("sum of split sizes must match split dim");
  }
  return outputs;
}

Variable tile(const Variable& input, const af::dim4& dims) {
  af::array result = af::tile(input.array(), dims);
  af::dim4 idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(
            sumAs(gradOutput, idims).array().as(inputs[0].type()), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sum(const Variable& input, const std::vector<int>& axes) {
  auto result = FL_ADJUST_INPUT_TYPE(input.array());
  for (size_t i = 0; i < axes.size(); i++) {
    result = af::sum(result, axes[i]);
  }
  af::dim4 indims = input.dims();
  auto gradFunc =
      [indims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(tileAs(gradOutput, indims).array(), false));
      };
  return Variable(result.as(input.type()), {input.withoutData()}, gradFunc);
}

Variable mean(const Variable& input, const std::vector<int>& axes) {
  auto result = FL_ADJUST_INPUT_TYPE(input.array());
  for (size_t i = 0; i < axes.size(); i++) {
    result = mean(result, axes[i]);
  }
  af::dim4 idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        af::dim4 odims = gradOutput.dims();
        dim_t count = 1;
        for (int i = 0; i < 4; i++) {
          count *= idims[i] / odims[i];
        }
        inputs[0].addGrad(
            Variable((tileAs(gradOutput, idims) / count).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable var(
    const Variable& in,
    const std::vector<int>& axes,
    const bool isbiased /* = false */) {
  auto input = FL_ADJUST_INPUT_TYPE(in);

  auto result = sum(input * input, axes);
  auto avg = mean(input, axes);
  auto n = 1;
  for (auto ax : axes) {
    n *= input.dims(ax);
  }
  if (!isbiased && n == 1) {
    throw std::invalid_argument(
        "cannot compute unbiased variance with only one sample");
  }
  auto val = 1.0 / (isbiased ? n : n - 1);
  result = val * (result - n * avg * avg);
  auto gradFunc =
      [val, axes](std::vector<Variable>& inputs, const Variable& gradOutput) {
        af::dim4 tiledims(1, 1, 1, 1);
        for (auto ax : axes) {
          tiledims[ax] = inputs[0].dims(ax);
        }
        inputs[0].addGrad(Variable(
            (2 * val * tile(gradOutput, tiledims) *
             (inputs[0] - tile(mean(inputs[0], axes), tiledims)))
                .array(),
            false));
      };
  return Variable(result.array(), {input}, gradFunc);
}

Variable
norm(const Variable& input, const std::vector<int>& axes, double p /* = 2 */) {
  if (p <= 0) {
    throw std::out_of_range("Lp norm: p must be > 0");
  }
  auto result = af::pow(af::abs(FL_ADJUST_INPUT_TYPE(input.array())), p);
  for (size_t i = 0; i < axes.size(); i++) {
    result = af::sum(result, axes[i]);
  }
  af::array sumap = result;
  result = af::pow(result, 1 / p);
  result.eval();

  auto gradFunc =
      [sumap, p](std::vector<Variable>& inputs, const Variable& gradOutput) {
        // correct, but less precise: auto gvar = Variable(af::pow(result, p-1),
        // false);
        auto gvar = Variable(af::pow(sumap, 1 - 1 / p), false);
        inputs[0].addGrad(Variable(
            (inputs[0] * fl::pow(fl::abs(inputs[0]), p - 2) *
             tileAs(gradOutput / gvar, inputs[0]))
                .array(),
            false));
      };
  return Variable(result, {input}, gradFunc);
}

Variable normalize(
    const Variable& in,
    const std::vector<int>& axes,
    double p /* = 2 */,
    double eps /* = 1e-12 */) {
  auto input = FL_ADJUST_INPUT_TYPE(in);
  Variable norm = fl::norm(input, axes, p);
  Variable invscale = max(norm, eps);
  return input / tileAs(invscale, input);
}

Variable matmul(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  // lhs:Input[0] -- [M, N]
  // rhs:Input[1] -- [N, K]
  // matmul(lhs, rhs)
  // -- matmul([M, N], [N, K]) --  [M, K]
  // result:gradOutput -- [M, K]
  auto result = matmul(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    if (inputs[0].isCalcGrad()) {
      // matmulNT(gradOutput, inputs[1])
      // -- matmulNT([M, K], [N, K])
      // -- matmul([M, K], [K, N]) -- [M, K]
      inputs[0].addGrad(
          Variable(matmulNT(gradOutput, inputs[1]).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      // matmulTN(inputs[0], gradOutput)
      // -- matmulTN([M, N], [M, K])
      // -- matmul([N, M], [M, K]) -- [N, K]
      inputs[1].addGrad(
          Variable(matmulTN(inputs[0], gradOutput).array(), false));
    }
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable matmulTN(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  // lhs:Input[0] -- [N, M]
  // rhs:Input[1] -- [N, K]
  // matmulTN(lhs, rhs)
  // -- matmulTN([N, M], [N, K])
  // -- matmul([M, N], [N, K]) -- [M, K]
  // result:gradOutput -- [M, K]
  auto result = matmulTN(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    if (inputs[0].isCalcGrad()) {
      // matmulNT(inputs[1], gradOutput)
      // -- matmulNT([N, K], [M, K])
      // -- matmul([N, K], [K, M]) -- [N, M]
      inputs[0].addGrad(
          Variable(matmulNT(inputs[1], gradOutput).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      // matmul(inputs[0], gradOutput)
      // -- matmulNT([N, M], [M, K]) -- [N, K]
      inputs[1].addGrad(Variable(matmul(inputs[0], gradOutput).array(), false));
    }
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable matmulNT(const Variable& lhs, const Variable& rhs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(lhs, rhs);
  // lhs:Input[0] -- [M, N]
  // rhs:Input[1] -- [K, N]
  // matmulNT(lhs, rhs)
  // -- matmulNT([M, N], [K, N])
  // -- matmul([M, N], [N, K]) -- [M, K]
  // result:gradOutput -- [M, K]
  auto result = matmulNT(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    if (inputs[0].isCalcGrad()) {
      // matmul(gradOutput, inputs[1])
      // -- matmul([M, K], [K, N]) -- [M, N]
      inputs[0].addGrad(Variable(matmul(gradOutput, inputs[1]).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      // matmulTN(gradOutput, inputs[0])
      // -- matmulTN([M, K], [M, N])
      // -- matmul([K, M], [M, N]) -- [K, N]
      inputs[1].addGrad(
          Variable(matmulTN(gradOutput, inputs[0]).array(), false));
    }
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable abs(const Variable& input) {
  auto result = af::abs(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    // af::sign returns signbit
    // Convert it into -1, 1
    auto sign = Variable(1 - 2 * af::sign(inputs[0].array()), false);
    inputs[0].addGrad(Variable((sign * gradOutput).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable flat(const Variable& input) {
  auto result = af::flat(input.array());
  af::dim4 idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(moddims(gradOutput, idims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable moddims(const Variable& input, const af::dim4& dims) {
  af::dim4 inferDims = dims;
  // Infer any 0 dim
  for (int i = 0; i < 4; ++i) {
    if (inferDims[i] == 0) {
      inferDims[i] = input.dims(i);
    }
  }
  // Infer any -1 dim
  int nInfer = 0;
  for (int i = 0; i < 4; i++) {
    if (inferDims[i] == -1) {
      nInfer++;
      inferDims[i] = -(input.elements() / inferDims.elements());
    }
  }

  if (nInfer > 1) {
    throw std::invalid_argument("too many dimensions for moddims to infer");
  }

  if (inferDims.elements() != input.elements()) {
    throw std::invalid_argument("mismatched # of elements in moddims");
  }

  auto result = af::moddims(input.array(), inferDims);

  af::dim4 inDims = input.dims();
  auto gradFunc =
      [inDims](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(moddims(gradOutput, inDims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable softmax(const Variable& input, const int dim) {
  af::array inputArr = FL_ADJUST_INPUT_TYPE(input.array());
  auto maxvals = af::max(inputArr, dim);
  af::dim4 tiledims(1, 1, 1, 1);
  tiledims[dim] = input.dims(dim);

  auto expInput = af::exp(inputArr - af::tile(maxvals, tiledims));
  auto result = expInput / af::tile(af::sum(expInput, dim), tiledims);

  result.eval();
  auto gradFunc = [dim, tiledims, result](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto rbyg = gradOutput.array() * result;
    auto gradSm = rbyg - result * af::tile(af::sum(rbyg, dim), tiledims);
    inputs[0].addGrad(Variable(gradSm.as(inputs[0].type()), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable logSoftmax(const Variable& input, const int dim) {
  af::array inputArr = FL_ADJUST_INPUT_TYPE(input.array());
  auto maxvals = max((inputArr), dim);
  af::dim4 tiledims(1, 1, 1, 1);
  tiledims[dim] = input.dims(dim);
  auto result = inputArr -
      af::tile(af::log(af::sum(
                   af::exp(inputArr - af::tile(maxvals, tiledims)), dim)) +
                   maxvals,
               tiledims);

  result.eval();
  auto gradFunc = [dim, tiledims, result](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto gradLsm = gradOutput.array() -
        af::exp(result) * af::tile(af::sum(gradOutput.array(), dim), tiledims);
    inputs[0].addGrad(Variable(gradLsm.as(inputs[0].type()), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable binaryCrossEntropy(const Variable& inputs, const Variable& targets) {
  auto targetsTyped = targets.as(inputs.type());
  return negate(
      targetsTyped * log(inputs) + (1 - targetsTyped) * log(1 - inputs));
}

Variable categoricalCrossEntropy(
    const Variable& in,
    const Variable& targets,
    ReduceMode reduction /* =ReduceMode::MEAN */,
    int ignoreIndex /* = -1 */) {
  auto input = FL_ADJUST_INPUT_TYPE(in);
  // input -- [C, X1, X2, X3]
  // target -- [X1, X2, X3, 1]
  for (int i = 1; i < 4; i++) {
    if (input.dims(i) != targets.dims(i - 1)) {
      throw std::invalid_argument(
          "dimension mismatch in categorical cross entropy");
    }
  }
  if (targets.dims(3) != 1) {
    throw std::invalid_argument(
        "dimension mismatch in categorical cross entropy");
  }

  int C = input.dims(0);
  int X = targets.elements();
  if (af::anyTrue<bool>(
          ((targets.array() < 0) || (targets.array() >= C)) &&
          (targets.array() != ignoreIndex))) {
    throw std::invalid_argument(
        "target contains elements out of valid range [0, num_categories) "
        "in categorical cross entropy");
  }

  auto x = af::moddims(input.array(), af::dim4(C, X));
  auto y = af::moddims(targets.array(), af::dim4(1, X));

  auto A = af::range(af::dim4(C, X));
  auto B = af::tile(y, af::dim4(C));
  auto mask = -(A == B); // [C X]

  auto result = mask * x;
  auto ignoreMask = af::flat(y == ignoreIndex); // [X, 1]
  result = af::flat(af::sum(result, 0)); // [X, 1]
  result(ignoreMask) = 0.;

  af::array denominator;
  if (reduction == ReduceMode::NONE) {
    result = af::moddims(result, targets.dims()); // [X1 X2 X3]
  } else if (reduction == ReduceMode::MEAN) {
    denominator = af::sum((!ignoreMask).as(af::dtype::s32), 0);
    result = af::sum(result, 0) / denominator; // [1]
  } else if (reduction == ReduceMode::SUM) {
    result = af::sum(result, 0); // [1]
  } else {
    throw std::invalid_argument(
        "unknown reduction method for categorical cross entropy");
  }

  auto inputDims = input.dims();
  auto gradFunc = [C, X, mask, ignoreMask, denominator, reduction, inputDims](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    af::array grad = gradOutput.array();
    if (reduction == ReduceMode::NONE) {
      grad = af::moddims(grad, af::dim4(X));
    } else if (reduction == ReduceMode::MEAN) {
      grad = af::tile(grad / denominator, af::dim4(X));
    } else if (reduction == ReduceMode::SUM) {
      grad = af::tile(grad, af::dim4(X));
    }
    // [1 X]
    grad(ignoreMask) = 0.;
    grad = af::moddims(grad, af::dim4(1, X));
    grad = af::tile(grad, af::dim4(C)) * mask;
    inputs[0].addGrad(Variable(af::moddims(grad, inputDims), false));
  };

  return Variable(result, {input.withoutData(), targets}, gradFunc);
}

Variable reorder(
    const Variable& input,
    const int dim0,
    const int dim1,
    const int dim2,
    const int dim3) {
  auto result = reorder(input.array(), dim0, dim1, dim2, dim3);
  if (!af::isLinear(result)) {
    auto tmp = af::array(result.dims(), input.type());
    af::copy(tmp, result, af::span);
    result = tmp;
  }

  std::vector<std::pair<int, int>> dimgrad = {
      {dim0, 0}, {dim1, 1}, {dim2, 2}, {dim3, 3}};
  std::sort(dimgrad.begin(), dimgrad.end());

  auto gradFunc =
      [dimgrad](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(Variable(
            reorder(
                gradOutput,
                dimgrad[0].second,
                dimgrad[1].second,
                dimgrad[2].second,
                dimgrad[3].second)
                .array(),
            false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable linear(const Variable& input, const Variable& weight) {
  auto dummyBias = Variable(af::array().as(input.type()), false);
  return linear(input, weight, dummyBias);
}

Variable linear(const Variable& in, const Variable& wt, const Variable& bs) {
  FL_VARIABLE_DTYPES_MATCH_CHECK(in, wt, bs);
  auto input = FL_ADJUST_INPUT_TYPE(in);
  auto weight = FL_ADJUST_INPUT_TYPE(wt);
  auto bias = FL_ADJUST_INPUT_TYPE(bs);

  af::dim4 to2d(input.dims(0), input.elements() / input.dims(0));
  auto to4d = input.dims();
  to4d[0] = weight.array().dims(0);

  auto output =
      moddims(af::matmul(weight.array(), moddims(input.array(), to2d)), to4d);

  auto hasBias = bias.elements() > 0;
  if (hasBias) {
    auto tiledims = output.dims();
    tiledims[0] = 1;
    output = output + tile(bias.array(), tiledims);
  }
  auto gradFunc = [hasBias](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto& in = inputs[0];
    auto& wt = inputs[1];

    af::array wtArray = wt.array();
    af::array gradOutputArray = gradOutput.array();

    auto nframes = in.elements() / in.dims(0);

    if (hasBias && inputs[2].isCalcGrad()) {
      auto& bs = inputs[2];
      bs.addGrad(Variable(sumAs(gradOutput, bs).array(), false));
    }
    if (in.isCalcGrad()) {
      af::dim4 to2dout(wtArray.dims(0), nframes);
      in.addGrad(Variable(
          moddims(matmulTN(wt, moddims(gradOutput, to2dout)), in.dims())
              .array(),
          false));
    }
    if (wt.isCalcGrad()) {
      af::dim4 to2din(wtArray.dims(1), nframes);
      af::dim4 to2dout(wtArray.dims(0), nframes);
      wt.addGrad(Variable(
          matmulNT(moddims(gradOutput, to2dout), moddims(in, to2din)).array(),
          false));
    }
  };
  if (hasBias) {
    return Variable(output, {input, weight, bias}, gradFunc);
  }
  return Variable(output, {input, weight}, gradFunc);
}

Variable gatedlinearunit(const Variable& input, const int dim) {
  auto inDims = input.dims();
  auto inType = input.type();
  auto inSize = inDims[dim];
  if (inSize % 2 == 1) {
    throw std::invalid_argument("halving dimension must be even for GLU");
  }

  std::array<af::seq, 4> fhalf, shalf;
  fhalf.fill(af::span);
  shalf.fill(af::span);
  fhalf[dim] = af::seq(inSize / 2);
  shalf[dim] = af::seq(inSize / 2, inSize - 1);
  af::array fhalfout = input.array()(fhalf[0], fhalf[1], fhalf[2], fhalf[3]);
  af::array shalfout = input.array()(shalf[0], shalf[1], shalf[2], shalf[3]);
  // Temporary workaround for indexing bug present in ArrayFire 3.6.1.
  fhalfout = af::moddims(fhalfout, fhalfout.dims());
  shalfout = af::moddims(shalfout, shalfout.dims());
  shalfout = af::sigmoid(shalfout);

  auto gradFunc = [fhalf, shalf, fhalfout, shalfout, inDims, inType](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto grad_glu = af::array(inDims, inType);
    grad_glu(fhalf[0], fhalf[1], fhalf[2], fhalf[3]) =
        shalfout * gradOutput.array();
    grad_glu(shalf[0], shalf[1], shalf[2], shalf[3]) =
        shalfout * (1.0 - shalfout) * fhalfout * gradOutput.array();
    inputs[0].addGrad(Variable(grad_glu, false));
  };
  return Variable(fhalfout * shalfout, {input.withoutData()}, gradFunc);
}

Variable embedding(const Variable& input, const Variable& embeddings) {
  if (input.numdims() >= 4) {
    throw std::invalid_argument("embedding input must have 3 or fewer dims");
  }

  auto idxs = af::flat(input.array());
  Variable result = Variable(embeddings.array()(af::span, idxs), false);

  auto inDims = input.dims();
  af::dim4 resultDims = {embeddings.dims(0), inDims[0], inDims[1], inDims[2]};

  result = moddims(result, resultDims);

  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    auto& w = inputs[1];
    if (!w.isCalcGrad()) {
      return;
    }

    auto ip = af::flat(inputs[0].array());
    auto deltas = af::moddims(gradOutput.array(), w.dims(0), ip.elements());

    auto sp = sparse(
        ip.elements(),
        w.dims(1),
        af::constant(1, ip.elements(), deltas.type()),
        af::range(af::dim4(ip.elements() + 1), 0, s32),
        ip.as(af::dtype::s32),
        AF_STORAGE_CSR);

    auto grad = transpose(matmulTN(sp, transpose(deltas)));
    w.addGrad(Variable(grad, false));
  };

  return Variable(result.array(), {input, embeddings}, gradFunc);
}

Variable padding(
    const Variable& input,
    std::vector<std::pair<int, int>> pad,
    double val) {
  af::dim4 opDims = input.dims();
  std::array<af::seq, 4> inSeq = {af::span, af::span, af::span, af::span};
  for (int i = 0; i < pad.size(); ++i) {
    opDims[i] += (pad[i].first + pad[i].second);
    inSeq[i] = af::seq(pad[i].first, opDims[i] - pad[i].second - 1);
  }
  af::array result = af::constant(val, opDims, input.type());
  result(inSeq[0], inSeq[1], inSeq[2], inSeq[3]) = input.array();

  auto gradFunc =
      [inSeq](std::vector<Variable>& inputs, const Variable& gradOutput) {
        inputs[0].addGrad(gradOutput(inSeq[0], inSeq[1], inSeq[2], inSeq[3]));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable dropout(const Variable& input, double p) {
  if (p > 0.0) {
    auto mask = Variable(
        (af::randu(input.dims(), input.type()) > p).as(input.type()), false);
    return 1.0 / (1.0 - p) * mask * input;
  } else {
    return input;
  }
}

Variable relu(const Variable& input) {
  return max(input, 0.0);
}

Variable gelu(const Variable& in) {
  auto input = FL_ADJUST_INPUT_TYPE(in);
  return 0.5 * input *
      (1.0 +
       fl::tanh(0.7978845608 * (input + 0.044715 * input * input * input)));
}

fl::Variable relativePositionEmbeddingRotate(const fl::Variable& input) {
  auto data = input.array();
  int d0 = data.dims(0);
  int d1 = data.dims(1);
  int d2 = data.dims(2);
  int d3 = data.dims(3);
  data = af::join(0, data, af::constant(0.0, d1, d1, d2, d3, data.type()));
  data = af::moddims(data, af::dim4((d0 + d1) * d1, 1, d2, d3));
  data = data.rows(0, (d1 + d0 - 1) * d1 - 1);
  data = af::moddims(data, af::dim4(d0 + d1 - 1, d1, d2, d3));
  auto gradFunc = [d0, d1, d2, d3](
                      std::vector<fl::Variable>& inputs,
                      const fl::Variable& gradOutput) {
    auto gradData = gradOutput.array();
    gradData = af::moddims(gradData, af::dim4((d0 + d1 - 1) * d1, 1, d2, d3));
    gradData = af::join(
        0, gradData, af::constant(0.0, d1, 1, d2, d3, gradData.type()));
    gradData = af::moddims(gradData, af::dim4(d0 + d1, d1, d2, d3));
    gradData = gradData.rows(0, d0 - 1);
    inputs[0].addGrad(fl::Variable(gradData, false));
  };
  return fl::Variable(data, {input}, gradFunc);
}

fl::Variable multiheadAttention(
    const fl::Variable& query,
    const fl::Variable& key,
    const fl::Variable& value,
    const fl::Variable& posEmb,
    const fl::Variable& mask,
    const int32_t nHeads,
    const double pDropout,
    const int32_t offset /* = 0 */) {
  int32_t bsz = query.dims(2);
  int32_t modelDim = query.dims(1);
  int32_t headDim = modelDim / nHeads;

  auto q = moddims(query, af::dim4(-1, headDim, nHeads * bsz));
  auto k = moddims(key, af::dim4(-1, headDim, nHeads * bsz));
  auto v = moddims(value, af::dim4(-1, headDim, nHeads * bsz));

  auto scores = matmulNT(q, k);
  if (!posEmb.isempty()) {
    int n = posEmb.dims(0) / 2 - offset;
    auto pscores = relativePositionEmbeddingRotate(matmulNT(posEmb.as(q.type()), q));
    scores = scores + transpose(pscores.rows(n, n + k.dims(0) - 1));
  }
  scores = scores / std::sqrt(float(headDim));
  if (!mask.isempty()) {
    scores = scores + tileAs(mask, scores);
  }

  auto attn = dropout(softmax(scores, 1), pDropout);
  auto result = matmul(attn.as(v.type()), v);
  result = moddims(result, af::dim4(-1, headDim * nHeads, bsz));
  return result;
}

} // namespace fl
