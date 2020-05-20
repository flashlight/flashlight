/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
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

#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"

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
      result = sum(result, i);
    }
  }
  return result;
}
} // namespace detail

Variable operator+(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() + rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(grad_output.array(), false));
    inputs[1].addGrad(Variable(grad_output.array(), false));
  };
  return Variable(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Variable operator+(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() + rhs_val;
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(grad_output.array(), false));
  };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator+(const double& lhs_val, const Variable& rhs) {
  return rhs + lhs_val;
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() - rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(grad_output.array(), false));
    inputs[1].addGrad(Variable(negate(grad_output).array(), false));
  };
  return Variable(result, {lhs.withoutData(), rhs.withoutData()}, gradFunc);
}

Variable operator-(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() - rhs_val;
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(grad_output.array(), false));
  };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator-(const double& lhs_val, const Variable& rhs) {
  auto result = lhs_val - rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(negate(grad_output).array(), false));
  };
  return Variable(result, {rhs.withoutData()}, gradFunc);
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() * rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    if (inputs[0].isCalcGrad()) {
      inputs[0].addGrad(Variable((grad_output * inputs[1]).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      inputs[1].addGrad(Variable((grad_output * inputs[0]).array(), false));
    }
  };
  return Variable(
      result,
      {rhs.isCalcGrad() ? lhs : lhs.withoutData(),
       lhs.isCalcGrad() ? rhs : rhs.withoutData()},
      gradFunc);
}

Variable operator*(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() * rhs_val;
  auto gradFunc =
      [rhs_val](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable((grad_output * rhs_val).array(), false));
      };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator*(const double& lhs_val, const Variable& rhs) {
  return rhs * lhs_val;
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() / rhs.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    auto inputs_1_rec = reciprocal(inputs[1]);
    auto grad_input_0 = grad_output * inputs_1_rec;
    if (inputs[0].isCalcGrad()) {
      inputs[0].addGrad(Variable(grad_input_0.array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      inputs[1].addGrad(Variable(
          (grad_input_0 * negate(inputs[0]) * inputs_1_rec).array(), false));
    }
  };
  return Variable(
      result, {rhs.isCalcGrad() ? lhs : lhs.withoutData(), rhs}, gradFunc);
}

Variable operator/(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() / rhs_val;
  auto gradFunc =
      [rhs_val](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable((grad_output / rhs_val).array(), false));
      };
  return Variable(result, {lhs.withoutData()}, gradFunc);
}

Variable operator/(const double& lhs_val, const Variable& rhs) {
  auto result = lhs_val / rhs.array();
  auto gradFunc = [lhs_val](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    inputs[0].addGrad(Variable(
        (grad_output * (-lhs_val) / (inputs[0] * inputs[0])).array(), false));
  };
  return Variable(result, {rhs}, gradFunc);
}

Variable operator>(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() > rhs.array();
  return Variable(result, false);
}

Variable operator>(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() > rhs_val;
  return Variable(result, false);
}

Variable operator>(const double& lhs_val, const Variable& rhs) {
  auto result = lhs_val > rhs.array();
  return Variable(result, false);
}

Variable operator<(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() < rhs.array();
  return Variable(result, false);
}

Variable operator<(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() < rhs_val;
  return Variable(result, false);
}

Variable operator<(const double& lhs_val, const Variable& rhs) {
  auto result = lhs_val < rhs.array();
  return Variable(result, false);
}

Variable operator>=(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() >= rhs.array();
  return Variable(result, false);
}

Variable operator>=(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() >= rhs_val;
  return Variable(result, false);
}

Variable operator>=(const double& lhs_val, const Variable& rhs) {
  auto result = lhs_val >= rhs.array();
  return Variable(result, false);
}

Variable operator<=(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() <= rhs.array();
  return Variable(result, false);
}

Variable operator<=(const Variable& lhs, const double& rhs_val) {
  auto result = lhs.array() <= rhs_val;
  return Variable(result, false);
}

Variable operator<=(const double& lhs_val, const Variable& rhs) {
  auto result = lhs_val <= rhs.array();
  return Variable(result, false);
}

Variable operator&&(const Variable& lhs, const Variable& rhs) {
  auto result = lhs.array() && rhs.array();
  return Variable(result, false);
}

Variable operator!(const Variable& input) {
  auto result = !input.array();
  return Variable(result, false);
}

Variable max(const Variable& lhs, const Variable& rhs) {
  auto result = max(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    auto mask = Variable(inputs[0].array() > inputs[1].array(), false);
    inputs[0].addGrad(Variable((mask * grad_output).array(), false));
    inputs[1].addGrad(Variable((!mask * grad_output).array(), false));
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable max(const Variable& lhs, const double& rhs_val) {
  auto result = max(lhs.array(), rhs_val);
  auto gradFunc =
      [rhs_val](std::vector<Variable>& inputs, const Variable& grad_output) {
        auto mask = Variable(inputs[0].array() > rhs_val, false);
        inputs[0].addGrad(Variable((mask * grad_output).array(), false));
      };
  return Variable(result, {lhs}, gradFunc);
}

Variable max(const double& lhs_val, const Variable& rhs) {
  return max(rhs, lhs_val);
}

Variable min(const Variable& lhs, const Variable& rhs) {
  auto result = min(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    auto mask = Variable(inputs[0].array() < inputs[1].array(), false);
    inputs[0].addGrad(Variable((mask * grad_output).array(), false));
    inputs[1].addGrad(Variable((!mask * grad_output).array(), false));
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable min(const Variable& lhs, const double& rhs_val) {
  auto result = min(lhs.array(), rhs_val);
  auto gradFunc =
      [rhs_val](std::vector<Variable>& inputs, const Variable& grad_output) {
        auto mask = Variable(inputs[0].array() < rhs_val, false);
        inputs[0].addGrad(Variable((mask * grad_output).array(), false));
      };
  return Variable(result, {lhs}, gradFunc);
}

Variable min(const double& lhs_val, const Variable& rhs) {
  return min(rhs, lhs_val);
}

Variable negate(const Variable& input) {
  auto result = 0.0 - input.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(negate(grad_output).array(), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable reciprocal(const Variable& input) {
  auto result = 1.0 / input.array();
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    auto res = reciprocal(inputs[0]);
    inputs[0].addGrad(
        Variable((negate(grad_output) * res * res).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable exp(const Variable& input) {
  auto result = exp(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable((grad_output * exp(inputs[0])).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable log(const Variable& input) {
  auto result = log(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable((grad_output / inputs[0]).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable log1p(const Variable& input) {
  auto result = log1p(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(
        Variable((grad_output / (1.0 + inputs[0])).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable sin(const Variable& input) {
  auto result = sin(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable((grad_output * cos(inputs[0])).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable cos(const Variable& input) {
  auto result = cos(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(
        Variable((grad_output * negate(sin(inputs[0]))).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable tanh(const Variable& input) {
  auto result = tanh(input.array());
  auto gradFunc =
      [result](std::vector<Variable>& inputs, const Variable& grad_output) {
        auto grad =
            Variable((1.0 - result * result) * grad_output.array(), false);
        inputs[0].addGrad(Variable(grad.array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable clamp(const Variable& input, const double lo, const double hi) {
  auto result = clamp(input.array(), lo, hi);
  auto gradFunc = [lo, hi, result](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    af::array grad_mask = grad_output.array();
    replace(grad_mask, (result > lo) && (result < hi), 0);
    inputs[0].addGrad(Variable(grad_mask, false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sqrt(const Variable& input) {
  auto result = af::sqrt(input.array());
  auto gradFunc = [result](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto output = Variable(result, false);
    inputs[0].addGrad(Variable((grad_output / (2 * output)).array(), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sigmoid(const Variable& input) {
  auto result = sigmoid(input.array());
  auto gradFunc =
      [result](std::vector<Variable>& inputs, const Variable& grad_output) {
        auto grad = grad_output.array() * result * (1 - result);
        inputs[0].addGrad(Variable(grad, false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable transpose(const Variable& input) {
  auto result = transpose(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    inputs[0].addGrad(Variable(transpose(grad_output).array(), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable tileAs(const Variable& input, const af::dim4& rdims) {
  auto result = detail::tileAs(input.array(), rdims);

  af::dim4 in_dims = input.dims();
  auto gradFunc =
      [in_dims](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable(sumAs(grad_output, in_dims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable tileAs(const Variable& input, const Variable& reference) {
  return tileAs(input, reference.dims());
}

Variable sumAs(const Variable& input, const af::dim4& rdims) {
  auto result = detail::sumAs(input.array(), rdims);
  auto idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable(tileAs(grad_output, idims).array(), false));
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
  auto dims = concatInputs[0].dims();
  int concat_size = dims[dim];
  for (int i = 1; i < concatInputs.size(); i++) {
    concat_size += concatInputs[i].dims(dim);
    for (int d = 0; d < 4; d++) {
      if (dim != d && concatInputs[i].dims(d) != dims[d]) {
        throw std::invalid_argument(
            "mismatch in dimension not being concatenated");
      }
    }
  }
  dims[dim] = concat_size;
  af::array result(dims, concatInputs[0].type());
  std::array<af::index, 4> slice{af::span, af::span, af::span, af::span};
  int start = 0;
  for (const auto& input : concatInputs) {
    slice[dim] = af::seq(start, start + input.dims(dim) - 1);
    result(slice[0], slice[1], slice[2], slice[3]) = input.array();
    start += input.dims(dim);
  }

  std::vector<Variable> inputs_nodata;
  std::vector<af::dim4> in_dims;

  for (const auto& in : concatInputs) {
    inputs_nodata.push_back(in.withoutData());
    in_dims.push_back(in.dims());
  }

  auto gradFunc = [dim, in_dims](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    std::array<af::index, 4> sx{af::span, af::span, af::span, af::span};
    int s = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      sx[dim] = af::seq(s, s + in_dims[i][dim] - 1);
      inputs[i].addGrad(
          Variable(grad_output.array()(sx[0], sx[1], sx[2], sx[3]), false));
      s += in_dims[i][dim];
    }
  };

  return Variable(result, inputs_nodata, gradFunc);
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
  auto result = tile(input.array(), dims);
  af::dim4 idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable(sumAs(grad_output, idims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable sum(const Variable& input, const std::vector<int>& axes) {
  auto result = input.array();
  for (size_t i = 0; i < axes.size(); i++) {
    result = sum(result, axes[i]);
  }
  af::dim4 indims = input.dims();
  auto gradFunc =
      [indims](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable(tileAs(grad_output, indims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable mean(const Variable& input, const std::vector<int>& axes) {
  auto result = input.array();
  for (size_t i = 0; i < axes.size(); i++) {
    result = mean(result, axes[i]);
  }
  af::dim4 idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& grad_output) {
        af::dim4 odims = grad_output.dims();
        dim_t count = 1;
        for (int i = 0; i < 4; i++) {
          count *= idims[i] / odims[i];
        }
        inputs[0].addGrad(
            Variable((tileAs(grad_output, idims) / count).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable var(
    const Variable& input,
    const std::vector<int>& axes,
    const bool isbiased /* = false */) {
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
      [val, axes](std::vector<Variable>& inputs, const Variable& grad_output) {
        af::dim4 tiledims(1, 1, 1, 1);
        for (auto ax : axes) {
          tiledims[ax] = inputs[0].dims(ax);
        }
        inputs[0].addGrad(Variable(
            (2 * val * tile(grad_output, tiledims) *
             (inputs[0] - tile(mean(inputs[0], axes), tiledims)))
                .array(),
            false));
      };
  return Variable(result.array(), {input}, gradFunc);
}

Variable norm(const Variable& input, const std::vector<int>& axes) {
  auto result = input.array() * input.array();
  for (size_t i = 0; i < axes.size(); i++) {
    result = sum(result, axes[i]);
  }
  result = af::sqrt(result);
  result.eval();

  auto gradFunc = [result](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto output = Variable(result, false);
    inputs[0].addGrad(Variable(
        (inputs[0] * tileAs(grad_output / output, inputs[0])).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable matmul(const Variable& lhs, const Variable& rhs) {
  // lhs:Input[0] -- [M, N]
  // rhs:Input[1] -- [N, K]
  // matmul(lhs, rhs)
  // -- matmul([M, N], [N, K]) --  [M, K]
  // result:grad_output -- [M, K]
  auto result = matmul(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    if (inputs[0].isCalcGrad()) {
      // matmulNT(grad_output, inputs[1])
      // -- matmulNT([M, K], [N, K])
      // -- matmul([M, K], [K, N]) -- [M, K]
      inputs[0].addGrad(
          Variable(matmulNT(grad_output, inputs[1]).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      // matmulTN(inputs[0], grad_output)
      // -- matmulTN([M, N], [M, K])
      // -- matmul([N, M], [M, K]) -- [N, K]
      inputs[1].addGrad(
          Variable(matmulTN(inputs[0], grad_output).array(), false));
    }
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable matmulTN(const Variable& lhs, const Variable& rhs) {
  // lhs:Input[0] -- [N, M]
  // rhs:Input[1] -- [N, K]
  // matmulTN(lhs, rhs)
  // -- matmulTN([N, M], [N, K])
  // -- matmul([M, N], [N, K]) -- [M, K]
  // result:grad_output -- [M, K]
  auto result = matmulTN(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    if (inputs[0].isCalcGrad()) {
      // matmulNT(inputs[1], grad_output)
      // -- matmulNT([N, K], [M, K])
      // -- matmul([N, K], [K, M]) -- [N, M]
      inputs[0].addGrad(
          Variable(matmulNT(inputs[1], grad_output).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      // matmul(inputs[0], grad_output)
      // -- matmulNT([N, M], [M, K]) -- [N, K]
      inputs[1].addGrad(
          Variable(matmul(inputs[0], grad_output).array(), false));
    }
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable matmulNT(const Variable& lhs, const Variable& rhs) {
  // lhs:Input[0] -- [M, N]
  // rhs:Input[1] -- [K, N]
  // matmulNT(lhs, rhs)
  // -- matmulNT([M, N], [K, N])
  // -- matmul([M, N], [N, K]) -- [M, K]
  // result:grad_output -- [M, K]
  auto result = matmulNT(lhs.array(), rhs.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    if (inputs[0].isCalcGrad()) {
      // matmul(grad_output, inputs[1])
      // -- matmul([M, K], [K, N]) -- [M, N]
      inputs[0].addGrad(
          Variable(matmul(grad_output, inputs[1]).array(), false));
    }
    if (inputs[1].isCalcGrad()) {
      // matmulTN(grad_output, inputs[0])
      // -- matmulTN([M, K], [M, N])
      // -- matmul([K, M], [M, N]) -- [K, N]
      inputs[1].addGrad(
          Variable(matmulTN(grad_output, inputs[0]).array(), false));
    }
  };
  return Variable(result, {lhs, rhs}, gradFunc);
}

Variable abs(const Variable& input) {
  auto result = af::abs(input.array());
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    // af::sign returns signbit
    // Convert it into -1, 1
    auto sign = Variable(1 - 2 * af::sign(inputs[0].array()), false);
    inputs[0].addGrad(Variable((sign * grad_output).array(), false));
  };
  return Variable(result, {input}, gradFunc);
}

Variable flat(const Variable& input) {
  auto result = af::flat(input.array());
  af::dim4 idims = input.dims();
  auto gradFunc =
      [idims](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable(moddims(grad_output, idims).array(), false));
      };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable moddims(const Variable& input, const af::dim4& dims) {
  af::dim4 infer_dims = dims;
  // Infer any 0 dim
  for (int i = 0; i < 4; ++i) {
    if (infer_dims[i] == 0) {
      infer_dims[i] = input.dims(i);
    }
  }
  // Infer any -1 dim
  int n_infer = 0;
  for (int i = 0; i < 4; i++) {
    if (infer_dims[i] == -1) {
      n_infer++;
      infer_dims[i] = -(input.elements() / infer_dims.elements());
    }
  }

  if (n_infer > 1) {
    throw std::invalid_argument("too many dimensions for moddims to infer");
  }

  if (infer_dims.elements() != input.elements()) {
    throw std::invalid_argument("mismatched # of elements in moddims");
  }

  auto result = af::moddims(input.array(), infer_dims);

  af::dim4 in_dims = input.dims();
  auto gradFunc = [in_dims](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    inputs[0].addGrad(Variable(moddims(grad_output, in_dims).array(), false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable softmax(const Variable& input, const int dim) {
  auto maxvals = max((input.array()), dim);
  af::dim4 tiledims(1, 1, 1, 1);
  tiledims[dim] = input.dims(dim);

  auto exp_input = exp(input.array() - tile(maxvals, tiledims));
  auto result = exp_input / tile(sum(exp_input, dim), tiledims);

  result.eval();
  auto gradFunc = [dim, tiledims, result](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto rbyg = grad_output.array() * result;
    auto grad_sm = rbyg - result * tile(sum(rbyg, dim), tiledims);
    inputs[0].addGrad(Variable(grad_sm, false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable logSoftmax(const Variable& input, const int dim) {
  auto maxvals = max((input.array()), dim);
  af::dim4 tiledims(1, 1, 1, 1);
  tiledims[dim] = input.dims(dim);
  auto result = input.array() -
      tile(log(sum(exp(input.array() - tile(maxvals, tiledims)), dim)) +
               maxvals,
           tiledims);

  result.eval();
  auto gradFunc = [dim, tiledims, result](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto grad_lsm = grad_output.array() -
        exp(result) * tile(sum(grad_output.array(), dim), tiledims);
    inputs[0].addGrad(Variable(grad_lsm, false));
  };
  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable binaryCrossEntropy(const Variable& inputs, const Variable& targets) {
  return negate(targets * log(inputs) + (1 - targets) * log(1 - inputs));
}

Variable categoricalCrossEntropy(
    const Variable& input,
    const Variable& targets,
    ReduceMode reduction /* =ReduceMode::MEAN */,
    int ignoreIndex /* = -1 */) {
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
  if (af::anyTrue<bool>((targets.array() < 0) || (targets.array() >= C))) {
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
  auto ignoreMask = (y != ignoreIndex).as(s32); // [1 X]
  result = ignoreMask * af::sum(result, 0); // [1 X]

  Variable denominator;
  if (reduction == ReduceMode::NONE) {
    result = af::moddims(result, targets.dims()); // [X1 X2 X3]
  } else if (reduction == ReduceMode::MEAN) {
    denominator = Variable(af::sum(ignoreMask, 1), false);
    result = af::sum(result, 1) / denominator.array(); // [1]
  } else if (reduction == ReduceMode::SUM) {
    result = af::sum(result, 1); // [1]
  } else {
    throw std::invalid_argument(
        "unknown reduction method for categorical cross entropy");
  }

  auto inputDims = input.dims();
  auto gradFunc = [C, X, mask, ignoreMask, denominator, reduction, inputDims](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto grad = gradOutput.array();
    if (reduction == ReduceMode::NONE) {
      grad = af::moddims(grad, af::dim4(1, X));
    } else if (reduction == ReduceMode::MEAN) {
      grad = af::tile(grad / denominator.array(), af::dim4(1, X));
    } else if (reduction == ReduceMode::SUM) {
      grad = af::tile(grad, af::dim4(1, X));
    }
    // [1 X]
    grad *= ignoreMask;
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
      [dimgrad](std::vector<Variable>& inputs, const Variable& grad_output) {
        inputs[0].addGrad(Variable(
            reorder(
                grad_output,
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
  auto dummy_bias = Variable(af::array(), false);
  return linear(input, weight, dummy_bias);
}

Variable
linear(const Variable& input, const Variable& weight, const Variable& bias) {
  auto has_bias = bias.elements() > 0;

  af::dim4 to2d(input.dims(0), input.elements() / input.dims(0));
  auto to4d = input.dims();
  to4d[0] = weight.dims(0);

  auto output =
      moddims(matmul(weight.array(), moddims(input.array(), to2d)), to4d);

  if (has_bias) {
    auto tiledims = output.dims();
    tiledims[0] = 1;
    output = output + tile(bias.array(), tiledims);
  }
  auto gradFunc = [has_bias](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto& in = inputs[0];
    auto& wt = inputs[1];
    auto nframes = in.elements() / in.dims(0);

    if (has_bias && inputs[2].isCalcGrad()) {
      auto& bs = inputs[2];
      bs.addGrad(Variable(sumAs(grad_output, bs).array(), false));
    }
    if (in.isCalcGrad()) {
      af::dim4 to2dout(wt.dims(0), nframes);
      in.addGrad(Variable(
          moddims(matmulTN(wt, moddims(grad_output, to2dout)), in.dims())
              .array(),
          false));
    }
    if (wt.isCalcGrad()) {
      af::dim4 to2din(wt.dims(1), nframes);
      af::dim4 to2dout(wt.dims(0), nframes);
      wt.addGrad(Variable(
          matmulNT(moddims(grad_output, to2dout), moddims(in, to2din)).array(),
          false));
    }
  };
  if (has_bias) {
    return Variable(output, {input, weight, bias}, gradFunc);
  }
  return Variable(output, {input, weight}, gradFunc);
}

Variable gatedlinearunit(const Variable& input, const int dim) {
  auto in_dims = input.dims();
  auto in_type = input.type();
  auto in_size = in_dims[dim];
  if (in_size % 2 == 1) {
    throw std::invalid_argument("halving dimension must be even for GLU");
  }

  std::array<af::seq, 4> fhalf, shalf;
  fhalf.fill(af::span);
  shalf.fill(af::span);
  fhalf[dim] = af::seq(in_size / 2);
  shalf[dim] = af::seq(in_size / 2, in_size - 1);
  af::array fhalfout = input.array()(fhalf[0], fhalf[1], fhalf[2], fhalf[3]);
  af::array shalfout = input.array()(shalf[0], shalf[1], shalf[2], shalf[3]);
  // Temporary workaround for indexing bug present in ArrayFire 3.6.1.
  fhalfout = af::moddims(fhalfout, fhalfout.dims());
  shalfout = af::moddims(shalfout, shalfout.dims());
  shalfout = af::sigmoid(shalfout);

  auto gradFunc = [fhalf, shalf, fhalfout, shalfout, in_dims, in_type](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    auto grad_glu = af::array(in_dims, in_type);
    grad_glu(fhalf[0], fhalf[1], fhalf[2], fhalf[3]) =
        shalfout * grad_output.array();
    grad_glu(shalf[0], shalf[1], shalf[2], shalf[3]) =
        shalfout * (1.0 - shalfout) * fhalfout * grad_output.array();
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

  auto in_dims = input.dims();
  af::dim4 result_dims = {
      embeddings.dims(0), in_dims[0], in_dims[1], in_dims[2]};

  result = moddims(result, result_dims);

  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& grad_output) {
    auto& w = inputs[1];
    if (!w.isCalcGrad()) {
      return;
    }

    auto ip = af::flat(inputs[0].array());
    auto deltas = af::moddims(grad_output.array(), w.dims(0), ip.elements());

    auto sp = sparse(
        ip.elements(),
        w.dims(1),
        af::constant(1, ip.elements(), deltas.type()),
        af::range(af::dim4(ip.elements() + 1), 0, s32),
        ip.as(s32),
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
  af::dim4 op_dims = input.dims();
  std::array<af::seq, 4> in_seq = {af::span, af::span, af::span, af::span};
  for (int i = 0; i < pad.size(); ++i) {
    op_dims[i] += (pad[i].first + pad[i].second);
    in_seq[i] = af::seq(pad[i].first, op_dims[i] - pad[i].second - 1);
  }
  af::array result = af::constant(val, op_dims, input.type());
  result(in_seq[0], in_seq[1], in_seq[2], in_seq[3]) = input.array();

  auto gradFunc = [in_seq](
                      std::vector<Variable>& inputs,
                      const Variable& grad_output) {
    inputs[0].addGrad(grad_output(in_seq[0], in_seq[1], in_seq[2], in_seq[3]));
  };

  return Variable(result, {input.withoutData()}, gradFunc);
}

Variable dropout(const Variable& input, double p) {
  if (p > 0.0) {
    auto mask = uniform(input.dims(), 0.0, 1.0, f32, false) > p;
    return (1.0 / (1.0 - p)) * mask * input;
  } else {
    return input;
  }
}

Variable relu(const Variable& input) {
  return max(input, 0.0);
}

Variable gelu(const Variable& input) {
  return 0.5 * input *
      (1.0 +
       fl::tanh(0.7978845608 * (input + 0.044715 * input * input * input)));
}

Variable constant(
    double val,
    int output_size,
    int input_size,
    af::dtype type,
    bool calc_grad) {
  return constant(val, af::dim4(output_size, input_size), type, calc_grad);
}

Variable constant(double val, af::dim4 dims, af::dtype type, bool calc_grad) {
  return Variable(af::constant(val, dims, type), calc_grad);
}

Variable
identity(int output_size, int input_size, af::dtype type, bool calc_grad) {
  return identity(af::dim4(output_size, input_size), type, calc_grad);
}

Variable identity(af::dim4 dims, af::dtype type, bool calc_grad) {
  return Variable(af::identity(dims, type), calc_grad);
}

Variable uniform(
    int output_size,
    int input_size,
    double min,
    double max,
    af::dtype type,
    bool calc_grad) {
  return uniform(af::dim4(output_size, input_size), min, max, type, calc_grad);
}

Variable
uniform(af::dim4 dims, double min, double max, af::dtype type, bool calc_grad) {
  af::array result = af::randu(dims, type);
  if (min != 0 || max != 1) {
    result = (max - min) * result + min;
  }
  return Variable(result, calc_grad);
}

Variable normal(
    int output_size,
    int input_size,
    double stdv,
    double mean,
    af::dtype type,
    bool calc_grad) {
  return normal(af::dim4(output_size, input_size), stdv, mean, type, calc_grad);
}

Variable normal(
    af::dim4 dims,
    double stdv,
    double mean,
    af::dtype type,
    bool calc_grad) {
  af::array result = af::randn(dims, type);
  if (mean != 0 || stdv != 1) {
    result = stdv * result + mean;
  }
  return Variable(result, calc_grad);
}

} // namespace fl
