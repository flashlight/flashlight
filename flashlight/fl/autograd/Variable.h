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

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include <arrayfire.h>

#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/common/Serialization.h"

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
 * fl::Variable aVar(af::randu(2), true); // Creates a variable
 * af::print("aVar", aVar.array());
 * // aVar
 * // [2 1 1 1]
 * //     0.6010
 * //     0.0278
 * auto bVar = 1.0 + log(aVar); // Perform some arithmetic operations
 * bVar.backward(); // Perform backward pass to compute the gradients
 * af::print("bVar Grad", bVar.grad().array());
 * // bVar Grad
 * // [2 1 1 1]
 * //    1.0000
 * //    1.0000
 * af::print("aVar Grad", aVar.grad().array());
 * // aVar Grad
 * // [2 1 1 1]
 * //    1.6640
 * //   36.0246
 * \endcode
 */
class Variable {
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
   * Creates a Variable which wraps the array specified
   * @param[in] data array to be stored in the Variable
   * @param[in] calcGrad specifies whether to the gradient is required for this
   * Variable
   */
  Variable(af::array data, bool calcGrad);

  /**
   * Creates a Variable which wraps the array and inputs specified
   * @param[in] data array to the stored in the Variable
   * @param[in] inputs a vector specifying inputs for this Variable
   * @param[in] gradFunc function specifying how to calculate gradient of the
   * input Variables
   */
  Variable(af::array data, std::vector<Variable> inputs, GradFunc gradFunc);

  /**
   * Indexing operator on the Arrayfire Array wrapped by the Variable
   * @param[in] s0 sequence of indices along first dimension
   * @param[in] s1 sequence of indices along second dimension
   * @param[in] s2 sequence of indices along third dimension
   * @param[in] s3 sequence of indices along fourth dimension
   * @param[in] unique boolean to specify whether all dimensions contain unique
   * indices
   * @return Variable storing the result after indexing operation
   */
  Variable operator()(
      const af::index& s0,
      const af::index& s1 = af::span,
      const af::index& s2 = af::span,
      const af::index& s3 = af::span,
      bool unique = false) const;

  /**
   * @return a reference to the underlying Arrayfire array.
   */
  af::array& array() const;

  /**
   * Creates a new variable based on the current variable whose type will be
   * adjusted based on the input type. Unlike `inPlaceCast`, `as` does not
   * change the current variable.
   *
   * @param[in] type target data type
   *
   * @return returns the casted variable.
   */
  Variable as(af::dtype type) const;

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
  af::dim4 dims() const;

  /**
   * Returns whether the array wrapped by the Variable is empty
   */
  bool isempty() const;

  /**
   * Returns whether the array wrapped by the Variable is contiguous in memory
   * in C order.
   */
  bool isLinear() const;

  /**
   * Returns a Variable with contiguous array containing the same data as self
   * array.
   */
  Variable linear() const;

  /**
   * Returns the type of the array wrapped by the Variable
   * (e.g. f32 for float, f64 for double).
   * Full list: http://arrayfire.org/docs/defines_8h.htm (search dtype)
   */
  af::dtype type() const;

  /**
   * Returns the total number of elements stored in array wrapped by the
   * Variable
   */
  dim_t elements() const;

  /**
   * Returns the total number of bytes stored in array wrapped by the
   * Variable
   */
  size_t bytes() const;

  /**
   * Returns the number of dimension of array wrapped by the Variable
   */
  unsigned numdims() const;

  /**
   * Returns the dimension of array wrapped by the Variable
   */
  dim_t dims(unsigned dim) const;

  /**
   * Evaluates any expressions in the array wrapped by the Variable
   */
  void eval() const;

  /**
   * Copies the array to the host and return the pointer.
   * Must eventually be freed with `af::freeHost()`.
   */
  template <typename T>
  T* host() const {
    return array().host<T>();
  }

  /**
   * Copies the array to the existing host pointer `ptr`
   */
  template <typename T>
  void host(T* ptr) const {
    array().host(ptr);
  }

  /**
   * Get the first element of the array as a scalar
   */
  template <typename T>
  T scalar() const {
    return array().scalar<T>();
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
   * Return a column from an array based on `index`. This can
   * also be seen as the result of doing
   * `input(af::span, index, af::span, af::span)`
   * @param[in] index index of the row
   * @return Variable storing the result
   */
  Variable col(int index) const;

  /**
   * Returns a sequence of columns from an array based on `first` and `last`
   * indices. This can also be seen as the result of doing
   * `input(af::span, af::seq(first, last), af::span, af::span)`
   * @param[in] first start index of the rows
   * @param[in] last end index of the rows
   * @return Variable storing the result
   */
  Variable cols(int first, int last) const;

  /**
   * Returns a row from an array based on `index`. This can
   * also be seen as the result of doing
   * `input(index, af::span, af::span, af::span)`
   * @param[in] index index of the slice
   * @return Variable storing the result
   */
  Variable row(int index) const;

  /**
   * Returns a sequence of rows from an array based on `first` and `last`
   * indices. This can also be seen as the result of doing
   * `input(af::seq(first, last), af::span, af::span, af::span)`
   * @param[in] first start index of the rows
   * @param[in] last end index of the rows
   * @return Variable storing the result
   */
  Variable rows(int first, int last) const;

  /**
   * Return slice in volume from an array based on `index`. This can
   * also be seen as the result of doing
   * `input(af::span, af::span, index, af::span)`
   * @param[in] index index of the slice
   * @return Variable storing the result
   */
  Variable slice(int index) const;

  /**
   * Return slices in volume from an array based on `first` and `last` indices.
   * This can also be seen as the result of doing `input(af::span, af::span,
   * af::seq(first, last), af::span)`
   * @param[in] first start index of the slices
   * @param[in] last end index of the slices
   * @return Variable storing the result
   */
  Variable slices(int first, int last) const;

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
    af::array data;

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

  std::shared_ptr<SharedData> sharedData_{std::make_shared<SharedData>()};
  std::shared_ptr<SharedGrad> sharedGrad_{std::make_shared<SharedGrad>()};

  // NB: array only; we don't try to serialize the autograd graph
  // Saving the sharedData ptr helps to avoid saving variables which share the
  // same underlying tensor twice
  FL_SAVE_LOAD(sharedData_, sharedGrad_)
};

} // namespace fl
