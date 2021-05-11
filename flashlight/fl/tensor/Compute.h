/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace af {
class array;
}

namespace fl {

/**
 * Block the calling host thread until all outstanding computation has
 * completed.
 *
 * The implementation of this function should synchronize any outstanding
 * computation abstractions, blocking accordingly.
 */
void sync();

// TODO:fl::Tensor {signature}
/**
 * Launches computation, [usually] asynchronously, on operations needed to make
 * a particular value for a tensor available.
 *
 * There is no implementation requirement for this function (it can be a noop),
 * but it may be called by the user in areas with empirically high memory
 * pressure or where computation might be prudent to perform as per the user's
 * discretion.
 *
 * @param[in] tensor the tensor on which to launch computation.
 */
void eval(af::array& tensor);

} // namespace fl
