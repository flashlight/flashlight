/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "flashlight/fl/autograd/Variable.h"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/common/Utils.h"
#include "flashlight/fl/tensor/Compute.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Shape.h"

namespace fl {

Variable::Variable(Tensor data, bool calcGrad) {
  sharedData_->data = std::move(data);
  sharedGrad_->calcGrad = calcGrad;
}

Variable::Variable(
    Tensor data,
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

Variable Variable::operator()(const std::vector<Index>& indices) const {
  auto result = tensor()(indices);
  auto inDims = shape();
  auto inType = type();

  auto gradFunc = [indices, inDims, inType](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    if (!inputs[0].isGradAvailable()) {
      auto grad = fl::full(inDims, 0.0, inType);
      inputs[0].addGrad(Variable(grad, false));
    }

    auto& grad = inputs[0].grad().tensor();
    grad(indices) += gradOutput.tensor();
  };
  return Variable(result, {this->withoutData()}, gradFunc);
}

Variable Variable::flat(const fl::Index& index) const {
  auto result = tensor().flat(index);
  auto inDims = shape();
  auto inType = type();

  auto gradFunc = [index, inDims, inType](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    if (!inputs[0].isGradAvailable()) {
      auto grad = fl::full(inDims, 0.0, inType);
      inputs[0].addGrad(Variable(grad, false));
    }
    auto& grad = inputs[0].grad().tensor();
    grad.flat(index) += gradOutput.tensor();
  };

  return Variable(result, {this->withoutData()}, gradFunc);
}

Tensor& Variable::tensor() const {
  return sharedData_->data;
}

Variable Variable::copy() const {
  return Variable(sharedData_->data, sharedGrad_->calcGrad);
}

Variable Variable::astype(fl::dtype newType) const {
  auto output = tensor().astype(newType);
  auto gradFunc = [](std::vector<Variable>& inputs,
                     const Variable& gradOutput) {
    auto& input = inputs[0];
    // Cast the grad output to match the type of the input's grad
    input.addGrad(Variable(gradOutput.tensor().astype(input.type()), false));
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

Shape Variable::shape() const {
  return tensor().shape();
}

bool Variable::isEmpty() const {
  return tensor().isEmpty();
}

bool Variable::isContiguous() const {
  return tensor().isContiguous();
}

Variable Variable::asContiguous() const {
  if (!isEmpty() && !isContiguous()) {
    tensor() = tensor().asContiguousTensor();
  }
  return *this;
}

fl::dtype Variable::type() const {
  return tensor().type();
}

Dim Variable::elements() const {
  return tensor().elements();
}

size_t Variable::bytes() const {
  return tensor().bytes();
}

unsigned Variable::ndim() const {
  return tensor().ndim();
}

Dim Variable::dim(unsigned dim) const {
  return tensor().dim(dim);
}

void Variable::eval() const {
  fl::eval(tensor());
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
         << childGrad.type() << " to a Variable of type " << this->type()
         << ". You might be performing an operation with "
            "two inputs of different types.";
      throw std::invalid_argument(ss.str());
    }
    if (childGrad.shape() != this->shape()) {
      std::stringstream ss;
      ss << "Variable::addGrad: given gradient has dimensions not equal "
            "to this Variable's dimensions: this variable has shape "
         << this->shape() << " whereas the child gradient has dimensions "
         << childGrad.shape() << std::endl;
      throw std::invalid_argument(ss.str());
    }
    if (sharedGrad_->grad) {
      // Prevent increment of array refcount to avoid a copy
      // if getting a device pointer. See
      // https://git.io/fp9oM for more
      sharedGrad_->grad = std::make_unique<Variable>(
          sharedGrad_->grad->tensor() + childGrad.tensor(), false);
    } else {
      // Copy the childGrad Variable so as to share a reference
      // to the underlying childGrad.tensor() rather than copying
      // the tensor into a new variable
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
  auto ones = Variable(fl::full(shape(), 1, this->type()), false);
  backward(ones, retainGraph);
}

Variable Variable::withoutData() const {
  Variable other;
  other.sharedGrad_ = sharedGrad_;
  // Ensure the type of the underlying [but empty] Tensor data is of the same
  // type and shape
  other.tensor() = Tensor(shape(), this->type());
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
