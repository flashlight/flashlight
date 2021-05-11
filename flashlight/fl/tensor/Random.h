/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {

/**
 * Sets the seed for a random number generator abstraction, if one exists. The
 * API expectation is that, barring hardware constraints, consistent RNG occurs
 * given some seed.
 *
 * @param[in] seed the seed to use
 */
void setSeed(int seed);

} // namespace fl
