/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Implements the forward part of 2D max pooling over input of dimentions
 * [ix, iy, chan, batch]. Results are written in to:
 * - ouput which is expected to be of dimensions [ox, oy, chan, batch].
 * - index which is expected to be of dimensions [ox * oy, chan, batch].
 * index output is the input to max_pool_bwd().
 * Suports sx>=px and sy>=py. Also assumes that px<=wx and py<=wy.
 */
void __kernel max_pool_fwd(
    __global float* input,
    __global float* output,
    __global unsigned int* index,
    int ix,
    int iy,
    int chan,
    int batch,
    int ox,
    int oy,
    int wx,
    int wy,
    int sx,
    int sy,
    int px,
    int py) {
  const int nWins = ox * oy;
  const int winNum = get_global_id(0);
  if (winNum >= nWins) return;
  const int c = get_global_id(1);
  if (c >= chan) return;
  const int b = get_global_id(2);
  if (b >= batch) return;

  // Window numbers are counted from 0 to ix * iy - 1 as column major.
  const int winX = winNum % ox;
  const int winY = winNum / ox;
  const int base = ix * iy * (c + b * chan);

  int xStart = (winX == 0) ? px : 0;
  int yStart = (winY == 0) ? py : 0;

  int xEnd = (winX == (ox - 1)) ? wx - px : wx;
  int yEnd = (winY == (oy - 1)) ? wy - py : wy;

  float mx = FLT_MIN;
  int mxOfst = -1;

  for (int y = yStart ; y < yEnd ; ++y) {
    for (int x = xStart ; x < xEnd ; ++x) {
      int xOfst = x + winX * sx - px;
      int yOfst = (y + winY * sy - py) * ix;
      int offset = xOfst + yOfst + base;

      float cur = input[offset];
      if (cur > mx) {
        mx = cur;
        mxOfst = offset;
      }
    }
  }

  const int oOfst = winX + ox * (winY + oy * (c + chan * b));
  output[oOfst] = mx;
  index[oOfst] = mxOfst;
}

/**
 * Implements the backward part of 2D max pooling over grad and ouput
 * with the same dimentions as the input argument to max_pool_fwd().
 * The index argument is the result of max_pool_fwd().
 * ouput is expected to be a tensor of all zeros.
 */
void __kernel max_pool_bwd(
    __global float* grad,
    __global float* output,
    __global unsigned int* index) {
  const int winNum = get_global_id(0);
  grad[index[winNum]] = output[winNum];
}
