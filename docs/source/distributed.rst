Distributed Training
====================

flashlight provides an easy-to-use API for distributed training. It uses `Facebook's Gloo library <https://github.com/facebookincubator/gloo>`_ when using the CPU backend, and `NVIDIA's NCCL library <https://developer.nvidia.com/nccl>`_ when using the CUDA backend. In the sections below, we briefly detail the API and document its use.

See ``examples/DistributedTraining.cpp`` for examples.

Setup
-----
To initialize the distributed environment, participating process must first perform an initial coordination step. flashlight supports multiple initialization methods, detailed below.

DistributedInit::MPI
####################

Use this initialization if spawning processes using `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_. MPI jobs are typically started from the command line using :code:`mpirun -n <NUM_PROC> [...]`. MPI will assign ranks for each process so that information can be broadcast for initial coordination.

::

  fl::distributedInit(
      fl::DistributedInit::MPI,
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Rank`
      -1, // worldRank - unused. Automatically derived from `MPI_Comm_Size`
      {}  // params
  );

DistributedInit::FILE_SYSTEM
############################

Use this initialization if all participating devices and processes have access to a shared filesystem. A shared file in that filesystem is used to initially coordinate participating processes. This shared file is specified via the ``fl::DistributedConstants::kFilePath`` key in the parameter map:

::

  fl::distributedInit(
      fl::DistributedInit::FILE_SYSTEM,
      [...], // worldRank. Each process calls with its corresponding rank.
      4, // worldSize
      {{fl::DistributedConstants::kFilePath, "/path/to/shared/filesystem/file"}});

When using the CUDA backend, ``fl::DistributedConstants::kMaxDevicePerNode`` must be passed as an additional required value in the parameter map to specify maximum number of GPU devices per node from which to derive a ``device-id``.

::

  std::cout << (int)fl::distributedBackend(); // 1 - fl::DistributedBackend::NCCL

  auto rank = fl::getWorldRank();
  std::cout << rank; // 0/1/2/3 depending on the process

  auto size = fl::getWorldSize();
  std::cout << size; // 4


Synchronizing Parameters
########################

Now, we demonstrate the implementation of a `data parallel <https://en.wikipedia.org/wiki/Data_parallelism>`_ model; during training, data is equally distributed amongst all devices, and each device completes full forward and backward passes independently, before synchronizing state via an ``allReduce`` operation. Below, we call ``allReduce`` on an ArrayFire array:

::

  auto a = af::constant(rank, 3, 3);
  fl::allReduce(a);
  af::print("a", a);
  // a
  // [3 3 1 1]
  //    6.0000     6.0000     6.0000
  //    6.0000     6.0000     6.0000
  //    6.0000     6.0000     6.0000

flashlight's distributed API also includes specific functions to synchronize ``Module`` parameters and register them for gradient synchronization. ``allReduceParameters`` synchronizes parameters of a ``Module`` across all processes (which is important in the case of random initialization), and ``distributeModuleGrads`` registers gradients in the ``Module`` for synchronization after each iteration of the backward pass:

::

  auto model = std::make_shared<Sequential>();
  // (add other modules to the Sequential)
  
  // synchronize parameters across processes
  fl::allReduceParameters(model);
  // add a hook to synchronize gradients of model parameters as they're computed
  fl::distributedModuleGrads(model, 1.0 / worldSize)
  // ...
  

Distributed Training
--------------------
In this section, we build on the :ref:`Perceptron training example <linearregression>` to run
data-parallel distributed training using synchronous `stochastic gradient descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_ (SGD).

First things first - initialize the distributed environment:
::

  // Uses MPI (Run with `mpirun -n 2`), CUDA backend
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

Create the dataset. Samples are divided equally among all processes.
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

Create a ``Module``, synchronize its parameters, and register gradients for synchronization:
::

  // Model defintion - 2-layer Perceptron with ReLU activation
  auto model = std::make_shared<Sequential>();
  model->add(Linear(nFeat, 100));
  model->add(ReLU());
  model->add(Linear(100, 1));
  // MSE loss
  auto loss = MeanSquaredError();

  // synchronize parameters across processes
  fl::allReduceParameters(model);

  // register gradients for synchronization
  fl::distributeModuleGrads(model, 1.0 / worldSize);

Create an ``Optimizer`` and ``Meter`` and start training:
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

On NVIDIA Tesla M40 GPUs, the above code runs in 3min 17sec while using distributed traininig with two GPUs, and runs in 5min 30sec without distributed training.
