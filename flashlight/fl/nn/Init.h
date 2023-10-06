/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"

/**
 * \defgroup nn_init_utils NN Initialization Functions
 *
 * Functions for initializing tensors.
 *
 * Provides facilities for creating a `fl::Variable` tensor of different types
 * and initializations vis-a-vis probability distributions, constants, and the
 * identity. Additionally wraps common tensors as integrated into modules.
 */

namespace fl {
namespace detail {

/**
 * Creates a `Tensor` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are distributed according to a uniform
 * distribution with parameters \f$\mathcal{U}(min, max)\f$. See [Uniform
 * Distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)).
 *
 * @param shape a Tensor shape
 * @param min the lower bound parameter for the uniform distribution
 * @param max the upper bound parameter for the uniform distribution
 * @param type the datatype for which to create the tensor
 *
 * @return a `Tensor` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Tensor uniform(
    const Shape& shape,
    double min = 0,
    double max = 1,
    fl::dtype type = fl::dtype::f32);

/**
 * Creates a `Tensor` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are distributed according to a  normal
 * distribution with parameters \f$\mathcal{N}(\mu, \sigma^2)\f$. See [Normal
 * Distribution](https://en.wikipedia.org/wiki/Normal_distribution).
 *
 * @param shape a Tensor shape
 * @param stdv the standard deviation by which to parameterize the distribution
 * @param mean the mean by which to parameterize the distribution
 * @param type the datatype for which to create the tensor
 *
 * @return a `Tensor` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Tensor normal(
    const Shape& shape,
    double stdv = 1,
    double mean = 0,
    fl::dtype type = fl::dtype::f32);

/**
 * Creates a `Tensor` representing a tensor with given input dimensions where
 * elements are uniformly distributed according to the method outlined in [He et
 * al (2015)](https://arxiv.org/abs/1502.01852): _Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification._
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Tensor
kaimingUniform(const Shape& shape, int fanIn, fl::dtype type = fl::dtype::f32);
/**
 * Creates a `Tensor` representing a tensor with given input dimensions
 * where elements are normally distributed according to the method outlined in
 * [He et al (2015)](https://arxiv.org/abs/1502.01852): _Delving Deep into
 * Rectifiers: Surpassing Human-Level Performance on ImageNet Classification._
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param type the datatype for which to create the tensor
 *
 * @return a `Tensor` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Tensor
kaimingNormal(const Shape& shape, int fanIn, fl::dtype type = fl::dtype::f32);

/**
 * Creates a `Tensor` representing a tensor with given input dimensions
 * where elements are uniformly distributed according to the method outlined in
 * [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural networks_.
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param fanOut number of output units in the Variable
 * @param type the datatype with which to create the tensor
 *
 * @return a `Tensor` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Tensor glorotUniform(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type = fl::dtype::f32);

/**
 * Creates a `Tensor` representing a tensor with given input dimensions
 * where elements are normally distributed according to the method outlined in
 * [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural
 * networks._
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param fanOut number of output units in the Variable
 * @param type the datatype for which to create the tensor
 *
 * @return a `Tensor` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Tensor glorotNormal(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type = fl::dtype::f32);

/*
 * Approximation of inverse error function.
 * TODO: to be removed once it's directly supported in AF.
 *
 * Implementation follows: https://git.io/JYuWA
 *
    Copyright (c) 2014 Indiana University

    Written by Prof. Gary L. Pavlis, Dept. of Geol. Sci.,
            Indiana University, Bloomington, IN
    This software is licensed under the New BSD license:
    Redistribution and use in source and binary forms,
    with or without modification, are permitted provided
    that the following conditions are met:
    Redistributions of source code must retain the above
    copyright notice, this list of conditions and the
    following disclaimer.
    Redistributions in binary form must reproduce the
    above copyright notice, this list of conditions and
    the following disclaimer in the documentation and/or
    other materials provided with the distribution.
    Neither the name of Indiana University nor
    the names of its contributors may be used to endorse
    or promote products derived from this software without
    specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
    CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
    WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
    THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
    USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
    IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
    USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
 */
FL_API Tensor erfinv(const Tensor& y);

} // namespace detail

/**
 * Constructs a `Variable` with gradient calculation disabled, from a given
 * array
 *
 * @param arr a `Tensor` to be used
 * @return a `Variable` from the given array with gradient calculation disabled
 *
 * \ingroup nn_init_utils
 */
FL_API Variable input(const Tensor& arr);

/**
 * See `fl::input` above.
 *
 * @param arr a `Tensor` to be used
 * @return a `Variable` from the given array with gradient calculation disabled
 *
 * \ingroup nn_init_utils
 */
FL_API Variable noGrad(const Tensor& arr);

/**
 * Constructs a `Variable` with gradient calculation enabled, from a given array
 *
 * @param arr a `Tensor` to be used
 * @return a `Variable` from the given array with gradient calculation enabled
 *
 * \ingroup nn_init_utils
 */
FL_API Variable param(const Tensor& arr);

/**
 * Creates a `Variable` representing a tensor with dimensions `[inputSize,
 * outputSize]` where all elements are a constant
 *
 * @param val the value of the constant in the tensor
 * @param inputSize the second dimension for the output tensor shape
 * @param outputSize the first dimension of the output tensor shape
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with constant values.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable constant(
    double val,
    int inputSize,
    int outputSize,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions where all elements are a constant
 *
 * @param val the value of the constant in the tensor
 * @param shape a Tensor shape
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with constant values.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable constant(
    double val,
    const Shape& shape,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a scalar with a given value and type.
 *
 * @param val the value of the tensor scalar
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with constant values.
 *
 * \ingroup nn_init_utils
 */
template <typename T>
Variable
scalar(T val, fl::dtype type = dtype_traits<T>::ctype, bool calcGrad = true) {
  return Variable(fromScalar(val, type), calcGrad);
}

/**
 * Creates a `Variable` representing an identity tensor with dimensions
 * `[inputSize, outputSize]`.
 *
 * @param inputSize the second dimension for the output tensor shape
 * @param outputSize the first dimension of the output tensor shape
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing the identity tensor.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable identity(
    int inputSize,
    int outputSize,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing an identity tensor of up to rank 4 with
 * arbitrary dimensions.
 *
 * @param shape a Tensor shape
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing the identity tensor.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable identity(
    const Shape& shape,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with dimensions `[inputSize,
 * outputSize]`, where elements are distributed according to a uniform
 * distribution with parameters \f$\mathcal{U}(min, max)\f$. See [Uniform
 * Distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)).
 *
 * @param inputSize the second dimension for the output tensor shape
 * @param outputSize the first dimension of the output tensor shape
 * @param min the lower bound parameter for the uniform distribution
 * @param max the upper bound parameter for the uniform distribution
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable uniform(
    int inputSize,
    int outputSize,
    double min = 0,
    double max = 1,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are distributed according to a uniform
 * distribution with parameters \f$\mathcal{U}(min, max)\f$. See [Uniform
 * Distribution](https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)).
 *
 * @param shape a Tensor shape
 * @param min the lower bound parameter for the uniform distribution
 * @param max the upper bound parameter for the uniform distribution
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable uniform(
    const Shape& shape,
    double min = 0,
    double max = 1,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with dimensions `[inputSize,
 * outputSize]` where elements are distributed according to a normal
 * distribution with parameters \f$\mathcal{N}(\mu, \sigma^2)\f$. See [Normal
 * Distribution](https://en.wikipedia.org/wiki/Normal_distribution).
 *
 * @param inputSize the second dimension for the output tensor shape
 * @param outputSize the first dimension of the output tensor shape
 * @param stdv the standard deviation by which to parameterize the distribution
 * @param mean the mean by which to parameterize the distribution
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable normal(
    int inputSize,
    int outputSize,
    double stdv = 1,
    double mean = 0,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are distributed according to a  normal
 * distribution with parameters \f$\mathcal{N}(\mu, \sigma^2)\f$. See [Normal
 * Distribution](https://en.wikipedia.org/wiki/Normal_distribution).
 *
 * @param shape a Tensor shape
 * @param stdv the standard deviation by which to parameterize the distribution
 * @param mean the mean by which to parameterize the distribution
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable normal(
    const Shape& shape,
    double stdv = 1,
    double mean = 0,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with given input dimensions where
 * elements are uniformly distributed according to the method outlined in [He et
 * al (2015)](https://arxiv.org/abs/1502.01852): _Delving Deep into Rectifiers:
 * Surpassing Human-Level Performance on ImageNet Classification._
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable kaimingUniform(
    const Shape& shape,
    int fanIn,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);
/**
 * Creates a `Variable` representing a tensor with given input dimensions where
 * elements are normally distributed according to the method
 * outlined in [He et al (2015)](https://arxiv.org/abs/1502.01852):
 * _Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification._
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable kaimingNormal(
    const Shape& shape,
    int fanIn,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with given input dimensions where
 * elements are uniformly distributed according to the
 * method outlined in [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural networks_.
 *
 * @param const shape& the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param fanOut number of output units in the Variable
 * @param type the datatype with which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable glorotUniform(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with given input dimensions where
 * elements are normally distributed according to the
 * method outlined in [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural
 * networks._
 *
 * @param shape the const shape& of output Variable
 * @param fanIn number of input units in the Variable
 * @param fanOut number of output units in the Variable
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable glorotNormal(
    const Shape& shape,
    int fanIn,
    int fanOut,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with given input dimensions where
 * elements are distributed according to the a truncated normal distribution as
 * in [here](https://en.wikipedia.org/wiki/Truncated_normal_distribution).
 *
 * @param shape an ArrayFire tensor shape
 * @param stdv the standard deviation by which to parameterize the distribution
 * @param mean the mean by which to parameterize the distribution
 * @param minCufOff the minimum value of the distribution
 * @param maxCutOff the maximum value of the distribution
 * @param type the datatype for which to create the tensor
 * @param calcGrad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 *
 * \ingroup nn_init_utils
 */
FL_API Variable truncNormal(
    const Shape& shape,
    double stdv = 1.,
    double mean = 0.,
    double minCufOff = -2.,
    double maxCutOff = 2.,
    fl::dtype type = fl::dtype::f32,
    bool calcGrad = true);

} // namespace fl
