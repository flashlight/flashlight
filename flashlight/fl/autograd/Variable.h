/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Serialization.h"
#include "flashlight/fl/tensor/TensorBase.h"

namespace fl {

/**
 *  Variable wraps an Arrayfire array and facilitates easy backpropagation
 *
 *  Variable is a wrapper around Arrayfire array and supports many operations
 * (Functions) on arrays. When a Function is applied on input Variable(s), the
 * output Variable(s) records it's inputs and a gradient function which
 * can be used to compute gradient propagated from output to each of its
 * inputs.
 *
 * Thus, Variable and Functions build a computation graph which is a DAG.
 * Following chain rule, backpropagation through this graph can be done easily
 * by traversing in a topologically sorted order starting from output
 * Variable(s).
 *
 * NOTE: Variable holds the underlying array and gradient as shared_ptrs.
 * Thus, calling the copy constructor of Variable creates a shallow copy
 * with the same underlying data!
 *
 * Example :
 *
 * \code{.cpp}
 * fl::Variable aVar(fl::rand({2}), true); // Creates a variable
 * std::cout << "aVar" << aVar.tensor());
 * // aVar
 * // [2 1 1 1]
 * //     0.6010
 * //     0.0278
 * auto bVar = 1.0 + log(aVar); // Perform some arithmetic operations
 * bVar.backward(); // Perform backward pass to compute the gradients
 * std::cout << "bVar Grad" << bVar.grad().tensor());
 * // bVar Grad
 * // [2 1 1 1]
 * //    1.0000
 * //    1.0000
 * std::cout << "aVar Grad" << aVar.grad().tensor());
 * // aVar Grad
 * // [2 1 1 1]
 * //    1.6640
 * //   36.0246
 * \endcode
 */
class FL_API Variable {
 public:
  using GradFunc = std::function<
      void(std::vector<Variable>& inputs, const Variable& grad_output)>;

  using GradHook = std::function<void(Variable& grad)>;

  /**
   * Creates an empty Variable. The underlying array is empty and
   * isCalcGrad() is false.
   */
  Variable() = default;

  /**
   * Creates a Variable which wraps the specified Tensor
   * @param[in] data Tensor to be stored in the Variable
   * @param[in] calcGrad specifies whether to the gradient is required for this
   * Variable
   */
  Variable(Tensor data, bool calcGrad);

  /**
   * Creates a Variable which wraps the specified Tensor and inputs
   * @param[in] data Tensor to the stored in the Variable
   * @param[in] inputs a vector specifying inputs for this Variable
   * @param[in] gradFunc function specifying how to calculate gradient of the
   * input Variables
   */
  Variable(Tensor data, std::vector<Variable> inputs, GradFunc gradFunc);

  Variable operator()(const std::vector<Index>& indices) const;

  /**
   * Indexing operator on a flattened Variable.
   * @param[in] indices a variable number of indices.
   * @return Variable storing the result after indexing operation
   */
  template <typename... Ts>
  Variable operator()(const Ts&... args) const {
    std::vector<Index> indices{{args...}};
    return this->operator()(indices);
  }

  /**
   * Indexing operator on a flattened Variable.
   * @param[in] index index with which to index the flattened tensor
   * @return Variable storing the result after indexing operation
   */
  Variable flat(const fl::Index& index) const;

  /**
   * @return a reference to the underlying Flashlight Tensor.
   */
  Tensor& tensor() const;

  /**
   * Creates a copy of this variable, but detached from the computation graph.
   * @return returns the cloned and detached variable.
   */
  Variable copy() const;

  /**
   * Creates a new variable based on the current variable whose type will be
   * adjusted based on the input type.
   *
   * @param[in] type target data type
   *
   * @return returns the casted variable.
   */
  Variable astype(fl::dtype type) const;

  /**
   * @return a reference to the underlying gradient Variable.
   */
  Variable& grad() const;

  /**
   * Returns whether the gradient calculation for the Variable is enabled
   */
  bool isCalcGrad() const;

  /**
   * Returns whether the gradient has been calculated for the Variable
   */
  bool isGradAvailable() const;

  /**
   * Returns the dimension of the array wrapped by the Variable
   */
  Shape shape() const;

  /**
   * Returns whether the array wrapped by the Variable is empty
   */
  bool isEmpty() const;

  /**
   * Returns whether the array wrapped by the Variable is contiguous in memory
   * in C order.
   */
  bool isContiguous() const;

  /**
   * Returns a Variable with contiguous array containing the same data as self
   * array.
   */
  Variable asContiguous() const;

  /**
   * Returns the type of the `Tensor` wrapped by the Variable
   * (e.g. f32 for float, f64 for double).
   *
   * See `fl/tensor/Types.h`.
   */
  fl::dtype type() const;

  /**
   * Returns the total number of elements stored in array wrapped by the
   * Variable
   */
  Dim elements() const;

  /**
   * Returns the total number of bytes stored in array wrapped by the
   * Variable
   */
  size_t bytes() const;

  /**
   * Returns the number of dimension of array wrapped by the Variable
   */
  unsigned ndim() const;

  /**
   * Returns the dimension of array wrapped by the Variable
   */
  Dim dim(unsigned dim) const;

  /**
   * Evaluates any expressions in the array wrapped by the Variable
   */
  void eval() const;

  /**
   * Copies the array to the host and return the pointer.
   * Must eventually be freed manually via `free` or a related call.
   */
  template <typename T>
  T* host() const {
    return tensor().host<T>();
  }

  /**
   * Copies the array to the existing host pointer `ptr`
   */
  template <typename T>
  void host(T* ptr) const {
    tensor().host(ptr);
  }

  /**
   * Get the first element of the array as a scalar
   */
  template <typename T>
  T scalar() const {
    return tensor().scalar<T>();
  }

  /**
   * Remove the gradient stored by the Variable
   */
  void zeroGrad();

  /**
   * Set whether to calculate gradient for the Variable.
   */
  void setCalcGrad(bool calcGrad);

  /**
   * Add the gradient `childGrad` to the Variable.
   * No-op if `this->isCalcGrad()` is false.
   */
  void addGrad(const Variable& childGrad);

  /**
   * Registers a lambda function `hook` to be applied on the gradient w.r.t
   * Variable after it is computed during backward pass
   */
  void registerGradHook(const GradHook& hook);

  /**
   * Clears the gradient hook stored in the variable
   */
  void clearGradHook();

  /**
   * Run backward pass on the Variable.  Gradient of all the inputs
   * in the computation graph leading up to the Variable on which the function
   * is computed.
   * @param[in] grad gradient w.r.t to the Variable
   * @param[in] retainGraph If False, clears the input Variables stored
   * by the Variable
   */
  void backward(const Variable& grad, bool retainGraph = false);

  /**
   * Run backward pass on the Variable. Gradient of all the inputs
   * in the computation graph leading up to the Variable on which the function
   * is computed. Gradient w.r.t the all the elements in the variable is set
   * to 1.0
   * @param[in] retainGraph If False, clears the input Variables stored
   * by the Variable
   */
  void backward(bool retainGraph = false);

  /**
   * Returns a copy of this variable after removing its underlying array.
   * The new Variable is used to store the inputs for a Variable
   * which doesn't need the output.
   */
  Variable withoutData() const;

 private:
  using DAG = std::vector<Variable>;

  /**
   * Get all the inputs to this Variable
   */
  std::vector<Variable>& getInputs() const;

  /**
   * Builds the computation graph which comprises of all the input Variables for
   * which the gradient of `var` can be propagated using chain rule
   */
  DAG build() const;

  /**
   * Calculate the gradient of inputs.
   * @param[in] retainGraph If False, clears the inputs stored
   * by the Variable
   */
  void calcGradInputs(bool retainGraph = false);

  /**
   * Calls the gradient hook (if any) registered by the Variable
   */
  void applyGradHook();

  struct SharedData {
    /// Array wrapped by this Variable
    Tensor data;

    FL_SAVE_LOAD(data)
  };

  struct SharedGrad {
    /// Whether the gradient should be computed for this Variable
    bool calcGrad{false};
    /// Inputs of this Variable
    std::vector<Variable> inputs;
    /// Gradient with respect to this Variable
    std::unique_ptr<Variable> grad{nullptr};
    /// Function for calculating the gradient of the input Variables
    GradFunc gradFunc{nullptr};
    /// Function applied to gradient after it's computed during bwd pass
    GradHook onGradAvailable{nullptr};

   private:
    FL_SAVE_LOAD(calcGrad);
  };

  std::shared_ptr<SharedData> sharedData_ = std::make_shared<SharedData>();
  std::shared_ptr<SharedGrad> sharedGrad_ = std::make_shared<SharedGrad>();

  // NB: array only; we don't try to serialize the autograd graph
  // Saving the sharedData ptr helps to avoid saving variables which share the
  // same underlying tensor twice
  FL_SAVE_LOAD(sharedData_, sharedGrad_)
};

} // namespace fl
