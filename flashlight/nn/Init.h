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
/**
 * @file nn/Init.h
 *
 * Functions for initializing tensors.
 *
 * Provides facilities for creating a `fl::Variable` tensor of different types
 * and initializations vis-a-vis probability distributions, constants, and the
 * identity. Additionally wraps common tensors as integrated into modules.
 */

#pragma once

#include "flashlight/autograd/Variable.h"
#include "flashlight/common/Defines.h"

namespace fl {

/**
 * Constructs a `Variable` with gradient calculation disabled, from a given
 * array
 *
 * @param arr an `af::array` to be used
 * @return a `Variable` from the given array with gradient calculation disabled
 */
Variable input(const af::array& arr);

/**
 * See `fl::input` above.
 *
 * @param arr an `af::array` to be used
 * @return a `Variable` from the given array with gradient calculation disabled
 */
Variable noGrad(const af::array& arr);

/**
 * Constructs a `Variable` with gradient calculation enabled, from a given array
 *
 * @param arr an `af::array` to be used
 * @return a `Variable` from the given array with gradient calculation enabled
 */
Variable param(const af::array& arr);

/**
 * Creates a `Variable` representing a tensor with dimensions `[input_size,
 * output_size]` where elements are uniformly distributed according to the
 * method outlined in [He et al (2015)](https://arxiv.org/abs/1502.01852):
 * _Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification._
 *
 * @param input_size the second dimension for the output tensor shape
 * @param output_size the first dimension of the output tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
//Variable kaimingUniform(
    //int input_size,
    //int output_size,
    //float gain = 1.0f,
    //af::dtype type = f32,
    //bool calc_grad = true,
    //bool fan_in = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are uniformly distributed according to the method
 * outlined in [He et al (2015)](https://arxiv.org/abs/1502.01852):
 * _Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification._
 *
 * @param dims an ArrayFire tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
Variable kaimingUniform(
    af::dim4 shape,
    int fanIn,
    float gain = 1.0f,
    af::dtype type = af::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor with dimensions `[input_size,
 * output_size]` where elements are normally distributed according to the method
 * outlined in [He et al (2015)](https://arxiv.org/abs/1502.01852):
 * _Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification._
 *
 * @param input_size the second dimension for the output tensor shape
 * @param output_size the first dimension of the output tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
Variable kaimingNormal(
    af::dim4 shape,
    int fanIn,
    float gain = 1.0f,
    af::dtype type = af::dtype::f32,
    bool calcGrad = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are normally distributed according to the method
 * outlined in [He et al (2015)](https://arxiv.org/abs/1502.01852):
 * _Delving Deep into Rectifiers: Surpassing Human-Level Performance on
 * ImageNet Classification._
 *
 * @param dims an ArrayFire tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
//Variable
//kaimingNormal(af::dim4 dims, af::dtype type = f32, bool calc_grad = true);

/**
 * Creates a `Variable` representing a tensor with dimensions `[input_size,
 * output_size]` where elements are uniformly distributed according to the
 * method outlined in [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural networks_.
 *
 * @param input_size the second dimension for the output tensor shape
 * @param output_size the first dimension of the output tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
Variable glorotUniform(
    int input_size,
    int output_size,
    af::dtype type = f32,
    bool calc_grad = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are uniformly distributed according to the
 * method outlined in [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural networks_.
 *
 * @param dims an ArrayFire tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
Variable
glorotUniform(af::dim4 dims, af::dtype type = f32, bool calc_grad = true);

/**
 * Creates a `Variable` representing a tensor with dimensions `[input_size,
 * output_size]` where elements are normally distributed according to the
 * method outlined in [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural
 * networks._
 *
 * @param input_size the second dimension for the output tensor shape
 * @param output_size the first dimension of the output tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
Variable glorotNormal(
    int input_size,
    int output_size,
    af::dtype type = f32,
    bool calc_grad = true);

/**
 * Creates a `Variable` representing a tensor of up to rank 4 with arbitrary
 * dimensions, where elements are normally distributed according to the
 * method outlined in [Glorot and Bengio
 * (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf):
 * _Understanding the difficulty of training deep feedforward neural
 * networks._
 *
 * @param dims an ArrayFire tensor shape
 * @param type the ArrayFire datatype for which to create the tensor
 * @param calc_grad flag denoting whether gradient calculation on the resulting
 * `Variable` should be enabled
 *
 * @return A `Variable` containing a tensor with random values distributed
 * accordingly.
 */
Variable
glorotNormal(af::dim4 dims, af::dtype type = f32, bool calc_grad = true);

} // namespace fl
