Distributed Training
====================

flashlight provides a easy-to-use API to perform distributed training. It  uses
`Gloo <https://github.com/facebookincubator/gloo>`_ for CPU backend and
`Nccl <https://developer.nvidia.com/nccl>`_ for Cuda backend. In this
tutorial, we will give a brief overview of it.

Setup
-----
The first step in setting up distributed environment is to initialize it so that all
the participating processes can perform the initial coordination step. flashlight supports
multiple initialization methods.

DistributedInit::MPI
####################

This initialization can be used if the processes are spawned using `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_.
These jobs are typically started using the command :code:`mpirun -n <NUM_PROC> [...]`.
MPI will be used to assign ranks for the processes and broadcast necessary information
for initial coordination steps.

::

  fl::distributedInit(
      fl::DistributedInit::MPI,
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Rank`
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Size`
      {} // param
  );

DistributedInit::FILE_SYSTEM
############################

This initialization can be used if all the participating processes have access to a
shared flle. The shared file will be used to setup the initial coordination step among
the processes.

::

  fl::distributedInit(
      fl::DistributedInit::FILE_SYSTEM,
      [...], // worldRank. Each process calls with its corresponding rank.
      4, // worldSize
      {{fl::DistributedConstants::kFilePath, "/path/to/shared/filesystem/file"}});

When using cuda backend, an additional param `fl::DistributedConstants::kMaxDevicePerNode`
is required which specifies maximum number of GPU devices per node to derive device-id.

::

  std::cout << (int)fl::distributedBackend(); // 1 - fl::DistributedBackend::NCCL

  auto rank = fl::getWorldRank();
  std::cout << rank; // 0/1/2/3 depending on the process

  auto size = fl::getWorldSize();
  std::cout << size; // 4


Now, we'll take a look at how to compute allReduce on an Arrayfire array.

::

  auto a = af::constant(rank, 3, 3);
  fl::allReduce(a);
  af::print("a", a);
  // a
  // [3 3 1 1]
  //    6.0000     6.0000     6.0000
  //    6.0000     6.0000     6.0000
  //    6.0000     6.0000     6.0000


Distributed Training
--------------------
In this section, we will extend the Perceptron training example to run
data-parallel distributed training using synchronous SGD.

First things first - initialize the distributed environment
::

  // Uses MPI (Run with `mpirun -n 2`) , CUDA backend
  fl::distributedInit(
      fl::DistributedInit::MPI,
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Rank`
      -1, // worldSize - unused. Automatically derived from `MPI_Comm_Size`
      {{fl::DistributedConstants::kMaxDevicePerNode, "8"}} // param
  );

  auto worldSize = fl::getWorldSize();
  auto worldRank = fl::getWorldRank();
  bool isMaster = (worldRank == 0);
  af::setSeed(worldRank);

Create the dataset. The samples are divided equally among all the processes.
::

  // Create dataset
  const int nSamples = 10000 / worldSize;
  const int nFeat = 10;
  auto X = af::randu(nFeat, nSamples) + 1; // X elements in [1, 2]
  auto Y = af::sum(af::pow(X, 3), 0).T() + // signal
           af::sin(2 * M_PI * af::randu(nSamples)); // noise
  // Create Dataset to simplify the code for iterating over samples
  TensorDataset data({X, Y});
  const int inputIdx = 0, targetIdx = 1;

Create the module and synchronize it's parameters. Also, registe

::

  // Model defintion - 2-layer Perceptron with ReLU activation
  auto model = std::make_shared<Sequential>();
  model->add(Linear(nFeat, 100));
  model->add(ReLU());
  model->add(Linear(100, 1));
  // MSE loss
  auto loss = MeanSquaredError();

  // synchronize parameters of the model so that the parameters in each process
  // is the same
  fl::allReduceParameters(model);

  // Add a hook to synchronize gradients of model parameters as they are
  // computed
  fl::distributeModuleGrads(model, 1.0 / worldSize);

Create Optimizer, Meter and run the training.
::

  // Optimizer definition
  const float learningRate = 0.0001;
  const float momentum = 0.9;
  auto sgd = SGDOptimizer(model->params(), learningRate, momentum);

  // Meter definition
  AverageValueMeter meter;

  // Start training

  if (isMaster) {
    std::cout << "[Multi-layer Perceptron] Started..." << std::endl;
  }
  const int nEpochs = 100;
  for (int e = 1; e <= nEpochs; ++e) {
    meter.reset();
    for (auto& sample : data) {
      sgd.zeroGrad();

      // Forward propagation
      auto result = model->forward(input(sample[inputIdx]));

      // Calculate loss
      auto l = loss(result, noGrad(sample[targetIdx]));

      // Backward propagation
      l.backward();

      // Update parameters
      sgd.step();

      meter.add(l.scalar<float>());
    }

    auto mse = meter.value();
    auto mseArr = af::array(1, &mse[0]);

    fl::allReduce(mseArr);
    if (isMaster) {
      std::cout << "Epoch: " << e << " Mean Squared Error: "
                << mseArr.scalar<double>() / worldSize << std::endl;
    }
  }
  if (isMaster) {
    std::cout << "[Multi-layer Perceptron] Done!" << std::endl;
  }
  // I1208 19:47:27.683432 3049001 DistributedBackend.cpp:190] Initialized NCCL successfully! Compiled with NCCL 2.2
  // [Multi-layer Perceptron] Started...
  // Epoch: 1 Mean Squared Error: 20.2124
  // Epoch: 2 Mean Squared Error: 5.28266
  // Epoch: 3 Mean Squared Error: 2.91948
  // Epoch: 4 Mean Squared Error: 2.50887
  // Epoch: 5 Mean Squared Error: 2.25293
  // ...
  // ...
  // ...
  // Epoch: 97 Mean Squared Error: 0.925514
  // Epoch: 98 Mean Squared Error: 0.922071
  // Epoch: 99 Mean Squared Error: 0.923678
  // Epoch: 100 Mean Squared Error: 0.922085
  // [Multi-layer Perceptron] Done!

The above code runs in 3min 17sec while using distributed traininig with 2 GPUs
while takes 5min 30sec without distributed training using Tesla M40 GPU(s).

Conclusion
----------

In this tutorial, we have shown how to flashlight to do distributed trainining.
All the source can be found in `examples/DistributedTraining.cpp`
