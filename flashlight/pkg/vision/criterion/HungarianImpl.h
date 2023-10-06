/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace fl {
namespace lib {
namespace set {

/*
 * Performs linear sum assignment
 * See (https://en.wikipedia.org/wiki/Assignment_problem) on a cost matrix
 * costs, of shape M X N, where we have M tasks and N workers
 * This function will write to rowIdxs and colIdxs * and expects
 * both to be of of size M. This has only been tested for N >  M
 * (We must have enough workers to do all tasks).
 * rowIdxs will contain the row idx for each assignment
 * and colIdxs wiill contain the colIdx for each assignment
 */
void hungarian(float* costs, int* rowIdxs, int* colIdxs, int M, int N);


/*
 * Same as above except it will output an M X N assignment matrix where
 * assignments[m][n] == 1 means m and n are assigned.
 */
 void hungarian(float* costs, int* assignments, int M, int N);

} // namespace set
} // namespace lib
} // namespace fl
