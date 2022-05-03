Extending flashlight
====================

Extending Modules
-----------------
flashlight provides a flexible API to describe new Modules so as to create complex neural architectures. Below, we detail creating ResNet 2-layer block by extending flashlight's ``Module``.

::

  #include <memory>

  #include "flashlight/fl/flashlight.h"

  class ResNetBlock : public fl::Container {
   public:
    explicit ResNetBlock(int channels = 2) {
      add(std::make_shared<fl::Conv2D>(
          channels, channels, 3, 3, 1, 1, fl::PaddingMode::SAME));
      add(std::make_shared<fl::Conv2D>(
          channels, channels, 3, 3, 1, 1, fl::PaddingMode::SAME));
    }

    // Custom forward pass
    std::vector<fl::Variable> forward(const std::vector<fl::Variable>& input) override {
      auto input = inputs[0];
      auto c1 = get(0);
      auto c2 = get(1);
      auto relu = fl::ReLU();
      auto out = relu(c1->forward(input));
      out = c2->forward(input) + input;
      return {relu(out)};
    }

    std::string prettyString() const override {
      return "2-Layer ResNetBlock Conv3x3";
    }

    template <class Archive>
    void serialize(Archive& ar) {
      ar(cereal::base_class<Container>(this));
    }
  };


Writing Custom Kernels
----------------------

While Flashlight backends such as ArrayFire provide fast Tensor operations, writing custom kernels is sometimes necessary for performance reasons. flashlight uses custom kernels with neural network accelerator libraries such as `mkl-dnn <https://github.com/intel/mkl-dnn>`_, `cuDNN <https://developer.nvidia.com/cudnn/>`_; others, such as `MIOpen <https://github.com/ROCmSoftwarePlatform/MIOpen>`_, can be easily wrapped.

flashlight makes this easy with a ``DevicePtr``, which gives raw pointers for underlying Flashlight Tensors enabling them to be operated on with APIs that read of write from raw pointers.

Here, we show an example of how one could use Baidu Research's `warp-ctc <https://github.com/baidu-research/warp-ctc>`_ to implement the `Connectionist Temporal Criterion <https://en.wikipedia.org/wiki/Connectionist_temporal_classification>`_  loss function.

::

  #include <vector>

  #include <ctc.h> // warp-ctc
  #include "flashlight/common/cuda.h" // cuda specific util. Not included in flashlight by default.
  #include "flashlight/fl/flashlight.h" // flashlight library

  fl::Variable ctc(const fl::Variable& input, const fl::Variable& target) {
    // Works only for batchsize = 1

    ctcOptions options;
    options.loc = CTC_GPU;
    options.stream = fl::cuda::getActiveStream();

    Tensor grad = fl::full(input.shape(), 0.0, input.type());

    int N = input.dim(0); // alphabet size
    int T = input.dim(1); // time frames
    int L = target.dim(0); // target length

    std::vector<int> inputLengths(T);
    size_t workspace_size;
    get_workspace_size(&L, inputLengths.data(), N, 1, options, &workspace_size);

    Tensor workspace({workspace_size}, fl::dtype::b8);

    float cost;
    {
      fl::DevicePtr inPtr(input.tensor());
      fl::DevicePtr gradPtr(grad);
      fl::DevicePtr wsPtr(workspace);
      int* labels = target.host<int>();
      compute_ctc_loss(
          (float*)inPtr.get(),
          (float*)gradPtr.get(),
          labels,
          &L,
          inputLengths.data(),
          N,
          1,
          &cost,
          wsPtr.get(),
          options);
      std::free(labels); // free host memory
    }
    Tensor result = Tensor::fromScalar(1, &cost);

    auto grad_func = [grad](
                         std::vector<fl::Variable>& inputs,
                         const fl::Variable& grad_output) {
      inputs[0].addGrad(fl::Variable(grad, false));
    };

    return fl::Variable(result, {input, target}, grad_func);
  }
