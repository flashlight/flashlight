Extending flashlight
====================

Extending Modules
-----------------
flashlight provides a flexible API to describe new modules. It is simple to extend
flashlight to describe complex neural architectures. Here, we show how one
can describe a ResNet 2-layer Block as a Module in flashlight.

::

  #include <memory>

  #include <flashlight/flashlight.h>

  class ResNetBlock : public fl::Container {
   public:
    explicit ResNetBlock(int channels = 2) {
      add(std::make_shared<fl::Conv2D>(
          channels, channels, 3, 3, 1, 1, fl::PaddingMode::SAME));
      add(std::make_shared<fl::Conv2D>(
          channels, channels, 3, 3, 1, 1, fl::PaddingMode::SAME));
    }

    // Custom forward pass
    fl::Variable forward(const fl::Variable& input) override {
      auto c1 = get(0);
      auto c2 = get(1);
      auto relu = fl::ReLU();
      auto out = relu(c1->forward(input));
      out = c2->forward(input) + input;
      return relu(out);
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

While the Arrayfire (tensor backend of flashlight) provides fast array operations,
there could still be cases where one would want to write custom kernels for
better performance or take advantage of NN Accelerator libraries like
`mkl-dnn <https://github.com/intel/mkl-dnn>`_, `cuDNN <https://developer.nvidia.com/cudnn/>`_,
`MIOpen <https://github.com/ROCmSoftwarePlatform/MIOpen>`_.

flashlight makes it easy to do this. Users can make use of `DevicePtr` class to
get device pointer of array and work on the pointers.

Here, we show an example of how one could use `warp-ctc <https://github.com/baidu-research/warp-ctc>`_
to implement `ConnectionistTemporalCriterion <https://en.wikipedia.org/wiki/Connectionist_temporal_classification>`_ Loss.

::

  #include <vector>

  #include <ctc.h> // warp-ctc
  #include <flashlight/common/cuda.h> // cuda specific util. Not included in flashlight by default.
  #include <flashlight/flashlight.h> // flashlight library

  fl::Variable ctc(const fl::Variable& input, const fl::Variable& target) {
    // Works only for batchsize = 1

    ctcOptions options;
    options.loc = CTC_GPU;
    options.stream = fl::cuda::getActiveStream();

    af::array grad = af::constant(0.0, input.dims(), input.type());

    int N = input.dims(0); // alphabet size
    int T = input.dims(1); // time frames
    int L = target.dims(0); // target length

    std::vector<int> inputLengths(T);
    size_t workspace_size;
    get_workspace_size(&L, inputLengths.data(), N, 1, options, &workspace_size);

    af::array workspace(workspace_size, af::dtype::b8);

    float cost;
    {
      fl::DevicePtr inPtr(input.array());
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
      af::freeHost(labels);
    }
    af::array result(1, &cost);

    auto grad_func = [grad](
                         std::vector<fl::Variable>& inputs,
                         const fl::Variable& grad_output) {
      inputs[0].addGrad(fl::Variable(grad, false));
    };

    return fl::Variable(result, {input, target}, grad_func);
  }
