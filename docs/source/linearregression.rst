Example: Linear Regression, Perceptron
======================================

In this tutorial, we will go through how to train a linear regression model
in flashlight. We will also extend it to train a simple Neural Network model,
two-layer Perceptron network to improve performance. We will also learn about
using :ref:`Modules<modules>`, :ref:`Datasets<datasets>`, :ref:`Meters<meters>`,
:ref:`Optimizers<optimizers>` classes which greatly simplifies the amount of  boilerplate
code that needs to be written for machine learning training.

This tutorial is meant to be a quick introduction to using flashlight for machine
learning related tasks. We do not use any validation sets to measure performance
for the sake of code simplicity.

Data creation
-------------

We will generate use artificially generated data for this tutorial. Input
consists of 10000 10D-samples generated randomly.

::

  const int nSamples = 10000;
  const int nFeat = 10;
  auto X = af::randu(nFeat, nSamples) + 1; // X elements in [1, 2]
  auto Y = /* signal */ af::sum(af::pow(X, 3), 0).T() +
      /* noise */ af::sin(2 * M_PI * af::randu(nSamples));

Linear Regression
-----------------

Initialize weight, bias and few other training params.

::

  const int nEpochs = 100;
  const float learningRate = 0.001;
  auto weight = fl::Variable(af::randu(1, nFeat), true /* isCalcGrad */);
  auto bias = fl::Variable(af::constant(0.0, 1), true /* isCalcGrad */);

Run linear regression using stochastic gradient descent

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

As you can see, Linear Regression model doesn't perform very well on this dataset
since target is non-linear transformation on input.

Multi-Layer Perceptron
----------------------

Now we'll train a model using Multi-Layer Perceptron. Unlike the previous
example on Linear Regression, we will use some abstraction in the training
pipeline using :ref:`Modules<modules>`, :ref:`Datasets<datasets>`, :ref:`Meters<meters>`,
:ref:`Optimizers<optimizers>` to greatly simplify the code. It is highly recommended
to use these abstractions to avoid any possible user errors.

Create `TensorDataset` to simplify the code for iterating over samples.

::

  TensorDataset data({X, Y});
  const int inputIdx = 0, targetIdx = 1;

The network is described using `Sequential` to easily keep track of params.

::

  // Model defintion - 2-layer Perceptron with ReLU activation
  Sequential model;
  model.add(Linear(nFeat, 100));
  model.add(ReLU());
  model.add(Linear(100, 1));
  // MSE loss
  auto loss = MeanSquaredError();

`SGDOptimizer` class helps to avoid writing all the standard parameter updating code for
Stochastic Gradient Descent .

::

  // Optimizer definition
  const float learningRate = 0.0001;
  const float momentum = 0.9;
  auto sgd = SGDOptimizer(model.params(), learningRate, momentum);

`AverageValueMeter` helps to keep track of metrics while training.

::

  // Meter definition
  AverageValueMeter meter;

Start training ...

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

Conclusion
----------

In this tutorial, we have shown how to flashlight to train very simple machine
learning models. All the source can be found in `examples/LinearRegression.cpp`,
`examples/Perceptron.cpp`,
