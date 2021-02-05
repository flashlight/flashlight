/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
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

void __kernel max_pool_bwd(
    __global float* grad,
    __global float* output,
    __global unsigned int* index) {
  const int winNum = get_global_id(0);
  grad[index[winNum]] = output[winNum];
}

