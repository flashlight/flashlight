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

#include "flashlight/fl/autograd/Variable.h"

#include <af/internal.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/common/Utils.h"

namespace fl {

Variable::Variable(af::array data, bool calcGrad) {
  sharedData_->data = std::move(data);
  sharedGrad_->calcGrad = calcGrad;
}

Variable::Variable(
    af::array data,
    std::vector<Variable> inputs,
    GradFunc gradFunc) {
  sharedData_->data = std::move(data);
  if (std::any_of(inputs.begin(), inputs.end(), [](const Variable& input) {
        return input.isCalcGrad();
      })) {
    sharedGrad_->calcGrad = true;
    sharedGrad_->inputs = std::move(inputs);
    sharedGrad_->gradFunc = std::move(gradFunc);
  }
}

Variable Variable::operator()(
    const af::index& idx1,
    const af::index& idx2, /* af::span */
    const af::index& idx3, /* af::span */
    const af::index& idx4, /* af::span */
    bool unique /* false */) const {
  // Get forward pass result using advanced indexing
  // from arrayfire
  auto result = array()(idx1, idx2, idx3, idx4);
  auto inDims = dims();
  auto inType = type();
  af::dim4 idxStart;
  af::dim4 idxEnd;
  std::vector<af::array> idxArr(4);
  bool advancedIndex = false;

  // Extract af::index variable information,
  // it can either be af::span, af::seq or af::array
  auto idxFunc = [&idxStart, &idxEnd, &idxArr, &advancedIndex, unique, inDims](
      const af::index& index, int pos) {
    if (index.isspan()) {
      idxStart[pos] = 0;
      idxEnd[pos] = inDims[pos];
    } else {
      const auto& idxSeq = index.get();
      if (idxSeq.isSeq) {
        // arrayfire uses inclusive last dimension, we use exclusive
        idxStart[pos] = idxSeq.idx.seq.begin;
        idxEnd[pos] = idxSeq.idx.seq.end + 1;
      } else {
        af_array arr;
        af_retain_array(&arr, idxSeq.idx.arr);
        idxArr[pos] = af::array(arr);
        idxStart[pos] = 0;
        idxEnd[pos] = idxArr[pos].dims(0);
        advancedIndex = !unique;
      }
    }
  };
  idxFunc(idx1, 0);
  idxFunc(idx2, 1);
  idxFunc(idx3, 2);
  idxFunc(idx4, 3);

  auto gradFunc =
      [idx1,
       idx2,
       idx3,
       idx4,
       idxStart,
       idxEnd,
       idxArr,
       advancedIndex,
       inDims,
       inType](std::vector<Variable>& inputs, const Variable& gradOutput) {
        if (!inputs[0].isGradAvailable()) {
          auto grad = af::constant(0.0, inDims, inType);
          inputs[0].addGrad(Variable(grad, false));
        }

        if (!advancedIndex) {
          auto& grad = inputs[0].grad().array();
          grad(idx1, idx2, idx3, idx4) += gradOutput.array();
        } else {
          gradAdvancedIndex(
              gradOutput, idxStart, idxEnd, inDims, idxArr, inputs[0].grad());
        }
      };
  return Variable(result, {this->withoutData()}, gradFunc);
}

af::array& Variable::array() const {
  return sharedData_->data;
}

Variable Variable::as(af::dtype newType) const {
  auto output = array().as(newType);
  auto gradFunc = [](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto& input = inputs[0];
    // Cast the grad output to match the type of the input's grad
    input.addGrad(Variable(gradOutput.array().as(input.type()), false));
  };
  return Variable(output, {this->withoutData()}, gradFunc);
}

Variable& Variable::grad() const {
  if (!sharedGrad_->calcGrad) {
    throw std::logic_error("gradient calculation disabled for this Variable");
  }

  if (!sharedGrad_->grad) {
    throw std::logic_error("gradient not calculated yet for this Variable");
  }

  return *sharedGrad_->grad;
}

std::vector<Variable>& Variable::getInputs() const {
  return sharedGrad_->inputs;
}

bool Variable::isCalcGrad() const {
  return sharedGrad_->calcGrad;
}

bool Variable::isGradAvailable() const {
  if (!sharedGrad_->calcGrad) {
    return false;
  }
  return sharedGrad_->grad != nullptr;
}

af::dim4 Variable::dims() const {
  return array().dims();
}

bool Variable::isempty() const {
  return array().isempty();
}

bool Variable::isLinear() const {
  return af::isLinear(array());
}

Variable Variable::linear() const {
  if (!isempty() && !isLinear()) {
    auto linearArray = af::array(dims(), type());
    af::copy(linearArray, array(), af::span);
    array() = linearArray;
  }
  return *this;
}

af::dtype Variable::type() const {
  return array().type();
}

dim_t Variable::elements() const {
  return array().elements();
}

size_t Variable::bytes() const {
  return array().bytes();
}

unsigned Variable::numdims() const {
  return array().numdims();
}

dim_t Variable::dims(unsigned dim) const {
  return array().dims(dim);
}

void Variable::eval() const {
  return array().eval();
}

void Variable::zeroGrad() {
  sharedGrad_->grad.reset();
}

void Variable::setCalcGrad(bool calcGrad) {
  sharedGrad_->calcGrad = calcGrad;
  if (!calcGrad) {
    sharedGrad_->gradFunc = nullptr;
    sharedGrad_->inputs.clear();
    sharedGrad_->grad.reset();
  }
}

void Variable::addGrad(const Variable& childGrad) {
  if (sharedGrad_->calcGrad) {
    // Ensure the type of the child grad is the same as the type of this
    // Variable (and transitively, that it's the same type as an existing grad)
    if (childGrad.type() != this->type()) {
      std::stringstream ss;
      ss << "Variable::addGrad: attempted to add child gradient of type "
         << afTypeToString(childGrad.type()) << " to a Variable of type "
         << afTypeToString(this->type())
         << ". You might be performing an operation with "
            "two inputs of different types.";
      throw std::invalid_argument(ss.str());
    }
    if (sharedGrad_->grad) {
      // Prevent increment of array refcount to avoid a copy
      // if getting a device pointer. See
      // https://git.io/fp9oM for more
      sharedGrad_->grad = std::make_unique<Variable>(
          sharedGrad_->grad->array() + childGrad.array(), false);
      // Eval the JIT as a temporary workaround for
      // https://github.com/arrayfire/arrayfire/issues/2281
      sharedGrad_->grad->eval();
    } else {
      // Copy the childGrad Variable so as to share a reference
      // to the underlying childGrad.array() rather than copying
      // the array into a new variable
      sharedGrad_->grad = std::make_unique<Variable>(childGrad);
    }
  }
}

void Variable::registerGradHook(const GradHook& hook) {
  sharedGrad_->onGradAvailable = hook;
}

void Variable::clearGradHook() {
  sharedGrad_->onGradAvailable = nullptr;
}

void Variable::applyGradHook() {
  if (sharedGrad_->onGradAvailable) {
    assert(sharedGrad_->grad);
    sharedGrad_->onGradAvailable(*sharedGrad_->grad);
  }
}

void Variable::calcGradInputs(bool retainGraph) {
  if (sharedGrad_->gradFunc) {
    if (!sharedGrad_->grad) {
      throw std::logic_error("gradient was not propagated to this Variable");
    }

    sharedGrad_->gradFunc(sharedGrad_->inputs, *sharedGrad_->grad);
  }
  if (!retainGraph) {
    sharedGrad_->inputs.clear();
  }
}

void Variable::backward(const Variable& grad, bool retainGraph) {
  addGrad(grad);
  auto dag = build();
  for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
    iter->calcGradInputs(retainGraph);
    iter->applyGradHook();
    if (!retainGraph) {
      *iter = Variable();
    }
  }
}

void Variable::backward(bool retainGraph) {
  auto ones = Variable(af::constant(1, dims(), this->type()), false);
  backward(ones, retainGraph);
}

Variable Variable::col(int index) const {
  auto result = array().col(index);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [index, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().col(index) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::cols(int first, int last) const {
  auto result = array().cols(first, last);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [first, last, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().cols(first, last) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::row(int index) const {
  auto result = array().row(index);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [index, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().row(index) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::rows(int first, int last) const {
  auto result = array().rows(first, last);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [first, last, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().rows(first, last) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::slice(int index) const {
  auto result = array().slice(index);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [index, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().slice(index) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::slices(int first, int last) const {
  auto result = array().slices(first, last);
  auto inDims = dims();
  auto inType = type();
  auto gradFunc = [first, last, inDims, inType](
      std::vector<Variable>& inputs, const Variable& gradOutput) {
    auto grad = Variable(af::constant(0, inDims, inType), false);
    grad.array().slices(first, last) = gradOutput.array();
    inputs[0].addGrad(grad);
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::withoutData() const {
  Variable other;
  other.sharedGrad_ = sharedGrad_;
  // Ensure the type of the underlying [but empty] data is the same; since
  // af::array is f32-initialized by default
  other.array() = af::array().as(this->type());
  return other;
}

Variable::DAG Variable::build() const {
  std::unordered_set<SharedGrad*> cache;
  DAG dag;
  std::function<void(const Variable&)> recurse;

  // Topological sort
  recurse = [&](const Variable& var) {
    auto id = var.sharedGrad_.get();
    if (cache.find(id) != cache.end()) {
      return;
    }
    for (const auto& input : var.getInputs()) {
      recurse(input);
    }
    cache.insert(id);
    dag.push_back(var);
  };

  recurse(*this);
  return dag;
}

} // namespace fl
