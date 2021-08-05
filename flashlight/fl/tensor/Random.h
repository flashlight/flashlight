/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "flashlight/fl/tensor/Shape.h"
#include "flashlight/fl/tensor/TensorBase.h"
#include "flashlight/fl/tensor/Types.h"

namespace fl {

/**
 * Sets the seed for a random number generator abstraction, if one exists. The
 * API expectation is that, barring hardware constraints, consistent RNG occurs
 * given some seed.
 *
 * @param[in] seed the seed to use
 */
void setSeed(int seed);

/**
 * Initialize a tensor with elements sampled from the standard normal
 * distribution.
 *
 * @param[in] dims the shape of the tensor to create
 * @param[in] type the type of the tensor. Defaults to float
 * @return a tensor with the given dimensions with elements sampled accordingly
 */
Tensor randn(const Shape& shape, dtype type = dtype::f32);

/**
 * Initialize a tensor with elements sampled uniformly from the interval [0, 1).
 *
 * @param[in] shape the shape of the tensor to create
 * @param[in] type the type of the tensor. Defaults to float
 * @return a tensor with the given dimensions with elements sampled accordingly
 */
Tensor rand(const Shape& shape, dtype type = dtype::f32);

} // namespace fl
