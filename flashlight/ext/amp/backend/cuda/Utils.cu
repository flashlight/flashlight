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
    int* __restrict isInvalidArrayPtr,
    int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    auto in = inputs[i];
    if (isnan(in) || isinf(in)) {
      *isInvalidArrayPtr = 1;
    }
  }
}

__global__ void scaleGradsKernel(
    float* __restrict grads,
    float* __restrict scaleFactorPtr,
    int* __restrict isInvalidArrayPtr,
    int size) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
    auto grad = grads[i];
    if ((*isInvalidArrayPtr) == 1) {
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

__global__ void decreaseScaleFactorKernel(
    float* __restrict scaleFactorPtr,
    int* __restrict isInvalidArrayPtr,
    float* __restrict minScaleFactorPtr) {
  if ((*isInvalidArrayPtr) == 1) {
    if ((*scaleFactorPtr) / 2 >= (*minScaleFactorPtr)) {
      *scaleFactorPtr = *scaleFactorPtr / 2;
    }
    *isInvalidArrayPtr = 0;
  }
}

__global__ void increaseScaleFactorKernel(
    float* __restrict scaleFactorPtr,
    float* __restrict maxScaleFactorPtr,
    int multiplicativeIncreasePtr) {
  if (multiplicativeIncreasePtr == 1) {
    if ((*scaleFactorPtr) * 2 <= (*maxScaleFactorPtr)) {
      *scaleFactorPtr = *scaleFactorPtr * 2;
    }
  } else {
    if ((*scaleFactorPtr) + 2 <= (*maxScaleFactorPtr)) {
      *scaleFactorPtr = *scaleFactorPtr + 2;
    }
  }
}

void validityCheck(af::array& in, af::array& isInvalidArray) {
  auto size = in.elements();
  in.eval();
  isInvalidArray.eval();
  auto dIn = in.device<float>();
  auto dIsInvalidArray = isInvalidArray.device<int>();
  uint numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;
  validityCheckKernel<<<
      numBlocks,
      NUM_THREADS,
      0,
      fl::cuda::getActiveStream()>>>(dIn, dIsInvalidArray, size);
  in.unlock();
  isInvalidArray.unlock();
}

void scaleGrads(af::array& grads, af::array& scaleFactor, af::array& isInvalidArray) {
  auto size = grads.elements();
  grads.eval();
  scaleFactor.eval();
  isInvalidArray.eval();
  auto dGrads = grads.device<float>();
  auto dScaleFactor = scaleFactor.device<float>();
  auto disInvalidArray = isInvalidArray.device<int>();
  uint numBlocks = (size + NUM_THREADS - 1) / NUM_THREADS;

  scaleGradsKernel<<<numBlocks, NUM_THREADS, 0, fl::cuda::getActiveStream()>>>(
      dGrads, dScaleFactor, disInvalidArray, size);
  grads.unlock();
  scaleFactor.unlock();
  isInvalidArray.unlock();
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

bool decreaseScaleFactor(
    af::array& scaleFactor,
    af::array& isInvalidArray,
    const af::array& minScaleFactor) {
  bool scaleIsValid = true;
  if (isInvalidArray.scalar<int>() == 1) {
    scaleIsValid = false;
  }
  scaleFactor.eval();
  isInvalidArray.eval();
  auto dScaleFactor = scaleFactor.device<float>();
  auto disInvalidArray = isInvalidArray.device<int>();
  auto dMinScaleFactor = minScaleFactor.device<float>();
  decreaseScaleFactorKernel<<<1, 1, 0, fl::cuda::getActiveStream()>>>(
      dScaleFactor, disInvalidArray, dMinScaleFactor);
  scaleFactor.unlock();
  isInvalidArray.unlock();
  minScaleFactor.unlock();
  return scaleIsValid;
}

void increaseScaleFactor(
    af::array& scaleFactor,
    const af::array& maxScaleFactor,
    const ScaleFactorIncreaseForm& increaseForm) {
  auto dScaleFactor = scaleFactor.device<float>();
  auto dMaxScaleFactor = maxScaleFactor.device<float>();
  int multiply = increaseForm == ScaleFactorIncreaseForm::MULTIPLICATIVE;
  increaseScaleFactorKernel<<<1, 1, 0, fl::cuda::getActiveStream()>>>(
      dScaleFactor, dMaxScaleFactor, multiply);
  scaleFactor.unlock();
  maxScaleFactor.unlock();
}
} // namespace ext
} // namespace fl
