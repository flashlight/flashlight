#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <af/cuda.h>
#include <arrayfire.h>

#include "flashlight/ext/amp/Utils.h"
#include "flashlight/fl/common/backend/cuda/CudaUtils.h"

#define NUM_THREADS 1024

namespace fl {
namespace ext {
__global__ void validityCheckKernel(
    float* __restrict inputs,
    int* __restrict flagPtr,
    int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    auto in = inputs[i];
    if (isnan(in) || isinf(in)) {
      *flagPtr = 1;
    }
  }
}

__global__ void scaleGradsKernel(
    float* __restrict grads,
    float* __restrict scaleFactorPtr,
    int* __restrict flagPtr,
    int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    auto grad = grads[i];
    if ((*flagPtr) == 1) {
      grad = 0;
    } else {
      grad = grad / (*scaleFactorPtr);
    }
    grads[i] = grad;
  }
}

__global__ void scaleLossKernel(
    float* __restrict loss,
    float* __restrict scaleFactorPtr,
    int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    loss[i] = loss[i] * (*scaleFactorPtr);
  }
}

__global__ void scaleLossKernelBackward(
    float* __restrict grad,
    float* __restrict scaleFactorPtr,
    float* __restrict upstreamGradPtr,
    int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    grad[i] = (*scaleFactorPtr) * (*upstreamGradPtr);
  }
}

__global__ void adjustScaleFactorKernel(
    float* __restrict scaleFactorPtr,
    int* __restrict flagPtr) {
  if ((*flagPtr) == 1) {
    *scaleFactorPtr = *scaleFactorPtr / 2;
    *flagPtr = 0;
  }
}

void validityCheck(af::array& in, af::array& flag) {
  auto size = in.elements();
  in.eval();
  flag.eval();
  auto dIn = in.device<float>();
  auto dFlag = flag.device<int>();
  uint numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;
  validityCheckKernel<<<
      numBlocks,
      NUM_THREADS,
      0,
      fl::cuda::getActiveStream()>>>(dIn, dFlag, size);
  in.unlock();
  flag.unlock();
}

void scaleGrads(af::array& grads, af::array& scaleFactor, af::array& flag) {
  auto size = grads.elements();
  grads.eval();
  scaleFactor.eval();
  flag.eval();
  auto dGrads = grads.device<float>();
  auto dScaleFactor = scaleFactor.device<float>();
  auto dFlag = flag.device<int>();
  uint numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;

  scaleGradsKernel<<<numBlocks, NUM_THREADS, 0, fl::cuda::getActiveStream()>>>(
      dGrads, dScaleFactor, dFlag, size);
  grads.unlock();
  scaleFactor.unlock();
  flag.unlock();
}

fl::Variable scaleLoss(fl::Variable& loss, fl::Variable& scaleFactor) {
  auto dims = loss.dims();
  auto size = loss.elements();
  loss.array().eval();
  scaleFactor.array().eval();
  auto dLoss = loss.array().device<float>();
  auto dScaleFactor = scaleFactor.array().device<float>();
  uint numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;
  scaleLossKernel<<<numBlocks, NUM_THREADS, 0, fl::cuda::getActiveStream()>>>(
      dLoss, dScaleFactor, size);
  loss.array().unlock();
  scaleFactor.array().unlock();

  auto gradFunc = [size, dims, scaleFactor](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto grad = af::array(dims);
    grad.eval();
    auto dGrad = grad.device<float>();
    auto dScaleFactor = scaleFactor.array().device<float>();
    auto dUpstreamGrad = gradOutput.array().device<float>();
    uint numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;
    scaleLossKernelBackward<<<
        numBlocks,
        NUM_THREADS,
        0,
        fl::cuda::getActiveStream()>>>(
        dGrad, dScaleFactor, dUpstreamGrad, size);
    grad.unlock();
    scaleFactor.array().unlock();
    gradOutput.array().unlock();
    inputs[0].addGrad(fl::Variable(grad, false));
  };
  return fl::Variable(loss.array(), {loss, scaleFactor}, gradFunc);
}

bool adjustScaleFactor(af::array& scaleFactor, af::array& flag) {
  scaleFactor.eval();
  flag.eval();
  auto dScaleFactor = scaleFactor.device<float>();
  auto dFlag = flag.device<int>();
  adjustScaleFactorKernel<<<1, 1, 0, fl::cuda::getActiveStream()>>>(
      dScaleFactor, dFlag);
  scaleFactor.unlock();
  flag.unlock();
  return true;
}
} // namespace ext
} // namespace fl
