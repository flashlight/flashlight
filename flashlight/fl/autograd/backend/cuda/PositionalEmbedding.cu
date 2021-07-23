#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include <arrayfire.h>

#include <flashlight/fl/common/backend/cuda/CudaUtils.h>

namespace fl {

namespace {
#define NUM_THREADS 1024
} // namespace

template <typename T>
__global__ void relativePositionalEmbeddingRotateKernel(
    const T* __restrict in,
    T* __restrict out,
    const int input_dim_0,
    const int output_dim_0,
    const int dim_1,
    const int dim_2) {
  const int featureMapId = blockIdx.x / dim_1;
  const int columnId = blockIdx.x % dim_1;

  const int inputFeatureMapSize = input_dim_0 * dim_1;
  const int outputFeatureMapSize = output_dim_0 * dim_1;
  const T* input = &in[featureMapId * inputFeatureMapSize];
  T* output = &out[featureMapId * outputFeatureMapSize];
  const int inputOffset = columnId * input_dim_0 - columnId;
  const int outputOffset = columnId * output_dim_0;
  for (int i = threadIdx.x; i < output_dim_0; i += blockDim.x) {
    T tmp = 0;
    if (i >= columnId && i < input_dim_0 + columnId) {
      tmp = input[inputOffset + i];
    }
    output[outputOffset + i] = tmp;
  }
}

af::array relativePositionalEmbeddingRotate(const af::array& input) {
  const int input_dim_0 = input.dims(0);
  const int dim_1 = input.dims(1);
  const int dim_2 = input.dims(2);
  const int output_dim_0 = input_dim_0 + dim_1 - 1;
  auto output = af::array(af::dim4(output_dim_0, dim_1, dim_2), input.type());
  input.eval();
  output.eval();

  if (input.type() == f16) {
    relativePositionalEmbeddingRotateKernel<half>
        <<<dim_1 * dim_2, NUM_THREADS, 0, fl::cuda::getActiveStream()>>>(
            input.device<half>(),
            output.device<half>(),
            input_dim_0,
            output_dim_0,
            dim_1,
            dim_2);
  } else if (input.type() == f32) {
    relativePositionalEmbeddingRotateKernel<float>
        <<<dim_1 * dim_2, NUM_THREADS, 0, fl::cuda::getActiveStream()>>>(
            input.device<float>(),
            output.device<float>(),
            input_dim_0,
            output_dim_0,
            dim_1,
            dim_2);
  } else {
    throw std::runtime_error("Unsupported Type in Position Embedding.");
  }
  input.unlock();
  output.unlock();
  return output;
}
} // namespace fl
