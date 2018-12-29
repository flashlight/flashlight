Example: Linear Regression, Perceptron
======================================

In this tutorial, we demonstrate how to train a simple linear regression model in flashlight. We then extend our implementation to a neural network vis-a-vis an implementation of a multi-layer perceptron to improve model performance. We show how to use :ref:`Modules<modules>`, :ref:`Datasets<datasets>`, :ref:`Meters<meters>`, and :ref:`Optimizers<optimizers>` in practice to reduce the amount of required boilerplate code for training.

This is meant to be a quick introduction to using flashlight for machine learning-related tasks; for the sake of simplicity, we do not use a validation set to measure performance.

Full code for the below examples can be found in ``examples/LinearRegression.cpp`` and ``examples/Perceptron.cpp``.

Data Creation
-------------

We artificially generate data for this tutorial. Our input consists of 10000 samples of randomly-generated 10D vectors.

::

  const int nSamples = 10000;
  const int nFeat = 10;
  auto X = af::randu(nFeat, nSamples) + 1; // X elements in [1, 2]
  auto Y = /* signal */ af::sum(af::pow(X, 3), 0).T() +
      /* noise */ af::sin(2 * M_PI * af::randu(nSamples));

Linear Regression
-----------------

First, we initialize our model parameters (weight and bias) along with the number of epochs in training, and the learning rate for our ``Optimizer``.

::

  const int nEpochs = 100;
  const float learningRate = 0.001;
  auto weight = fl::Variable(af::randu(1, nFeat), true /* isCalcGrad */);
  auto bias = fl::Variable(af::constant(0.0, 1), true /* isCalcGrad */);

Now, run linear regression using stochastic gradient descent:

::

  std::cout << "[Linear Regression] Started..." << std::endl;

  for (int e = 1; e <= nEpochs; ++e) {
    af::array error = af::constant(0, 1);
    for (int i = 0; i < nSamples; ++i) {
      auto input = fl::Variable(X(af::span, i), false /* isCalcGrad */);
      auto yPred = fl::matmul(weight, input) + bias;

      auto yTrue = fl::Variable(Y(i), false /* isCalcGrad */);

      // Mean Squared Error
      auto loss = ((yPred - yTrue) * (yPred - yTrue)) / nSamples;

      // Compute gradients using backprop
      loss.backward();

      // Update the weight and bias
      weight.array() = weight.array() - learningRate * weight.grad().array();
      bias.array() = bias.array() - learningRate * bias.grad().array();

      // clear the gradients for next iteration
      weight.zeroGrad();
      bias.zeroGrad();

      error += loss.array();
    }

    std::cout << "Epoch: " << e
              << " Mean Squared Error: " << error.scalar<float>() << std::endl;
  }

  std::cout << "[Linear Regression] Done!" << std::endl;

  // [LinearRegression] Started...
  // Epoch: 1 Mean Squared Error: 847.404
  // Epoch: 2 Mean Squared Error: 772.851
  // Epoch: 3 Mean Squared Error: 704.999
  // Epoch: 4 Mean Squared Error: 643.251
  // ...
  // ...
  // ..
  // Epoch: 97 Mean Squared Error: 18.4037
  // Epoch: 98 Mean Squared Error: 18.3947
  // Epoch: 99 Mean Squared Error: 18.3864
  // Epoch: 100 Mean Squared Error: 18.3789
  // [LinearRegression] Done!

Our regression model doesn't perform well on this dataset; ``target`` is the result of a non-linear transformation of the input.

Multi-Layer Perceptron
----------------------

Next, we implement and train a multi-layer perceptron. Here, we take advantage of abstractions on flashlight's training pipeline: :ref:`Modules<modules>`, :ref:`Datasets<datasets>`, :ref:`Meters<meters>`, and :ref:`Optimizers<optimizers>`, which greatly simplify our implementation and make it less error-prone.

Here, we create a `TensorDataset` to reduce boilerplate for iterating over samples:

::

  TensorDataset data({X, Y});
  const int inputIdx = 0, targetIdx = 1;

We compose components of the network into a `Sequential` container to easily keep track of parameters and take advantage of other abstractions:

::

  // Model defintion - 2-layer Perceptron with ReLU activation
  Sequential model;
  model.add(Linear(nFeat, 100));
  model.add(ReLU());
  model.add(Linear(100, 1));
  // MSE loss
  auto loss = MeanSquaredError();

We create an `SGDOptimizer`, which eliminates repetitive code for parameter updates during stochastic gradient descent:

::

  // Optimizer definition
  const float learningRate = 0.0001;
  const float momentum = 0.9;
  auto sgd = SGDOptimizer(model.params(), learningRate, momentum);

`AverageValueMeter` helps to keep track of metrics while training:

::

  // Meter definition
  AverageValueMeter meter;

We're ready to start training:

::

  std::cout << "[Multi-layer Perceptron] Started..." << std::endl;

  const int nEpochs = 100;
  for (int e = 1; e <= nEpochs; ++e) {
    meter.reset();
    for (auto& sample : DatasetIterator(&data)) {
      sgd.zeroGrad();

      // Forward propagation
      auto result = model(input(sample[inputIdx]));

      // Calculate loss
      auto l = loss(result, noGrad(sample[targetIdx]));

      // Backward propagation
      l.backward();

      // Update parameters
      sgd.step();

      meter.add(l.scalar<float>());
    }
    std::cout << "Epoch: " << e
              << " Mean Squared Error: " << meter.value()[0] << std::endl;
  }
  std::cout << "[Multi-layer Perceptron] Done!" << std::endl;

  // [Multi-layer Perceptron] Started...
  // Epoch: 1 Mean Squared Error: 13.13
  // Epoch: 2 Mean Squared Error: 2.58897
  // Epoch: 3 Mean Squared Error: 2.10619
  // Epoch: 4 Mean Squared Error: 1.84273
  // ..
  // ..
  // ..
  // Epoch: 97 Mean Squared Error: 0.817783
  // Epoch: 98 Mean Squared Error: 0.819474
  // Epoch: 99 Mean Squared Error: 0.8187
  // Epoch: 100 Mean Squared Error: 0.813558
  // [Multi-layer Perceptron] Done!
