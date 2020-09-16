#pragma once

namespace fl {
namespace cv {

void hungarian(float* costs, int* rowIdxs, int* colIdxs, int M, int N);

void hungarian(float* costs, int* assignments, int M, int N);

} // cv
} // flashlight
