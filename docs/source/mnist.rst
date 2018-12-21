Example: MNIST
==============

In this example we will show how to build a simple convolutional network and
train it on `MNIST <http://yann.lecun.com/exdb/mnist/>`_. Download and unpack
the dataset before running.

Data
----

The first step is loading the data

::

  array train_x;
  array train_y;
  std::tie(train_x, train_y) = load_dataset(data_dir);

  // Hold out a dev set
  auto val_x = train_x(span, span, seq(0, VAL_SIZE - 1));
  train_x = train_x(span, span, seq(VAL_SIZE, TRAIN_SIZE - 1));
  auto val_y = train_y(seq(0, VAL_SIZE - 1));
  train_y = train_y(seq(VAL_SIZE, TRAIN_SIZE - 1));

  // Make the training batch dataset
  BatchDataset trainset(
      std::make_shared<TensorDataset>(std::vector<af::array>{train_x, train_y}),
      batch_size);

  // Make the validation batch dataset
  BatchDataset valset(
      std::make_shared<TensorDataset>(std::vector<af::array>{val_x, val_y}),
      batch_size);

Model
-----

The next step is to construct the model.

::

  Sequential model;
  auto pad = PaddingMode::SAME;
  model.add(Conv2D(1 /* input channels */,
                   32 /* output channels */,
                   5 /* kernel width */,
                   5 /* kernel height */,
                   1 /* stride x */,
                   1 /* stride y */,
                   pad /* padding mode */));
  model.add(ReLU());
  model.add(Pool2D(2 /* kernel width */,
                   2 /* kernel height */,
                   2 /* stride x */,
                   2 /* stride y */));
  model.add(Conv2D(32, 64, 5, 5, 1, 1, pad));
  model.add(ReLU());
  model.add(Pool2D(2, 2, 2, 2));
  model.add(View({7 * 7 * 64, -1}));
  model.add(Linear(7 * 7 * 64, 1024));
  model.add(ReLU());
  model.add(Dropout(0.5));
  model.add(Linear(1024, 10));
  model.add(LogSoftmax());

Training
--------

First we make an optimizer, and then run the training loop for a specified
number of epochs (passes over the full dataset).

::

  // Make the optimizer
  SGDOptimizer opt(model.params(), learning_rate);

  // The main training loop
  for (int e = 0; e < epochs; e++) {
    AverageValueMeter train_loss_meter;

    // Get an iterator over the data
    for (auto& example : dataset) {
      auto inputs = noGrad(example[INPUT_IDX]);
      auto output = model(inputs);


      auto target = noGrad(example[TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      train_loss_meter.add(loss.array().scalar<float>());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();
      opt.step();
      opt.zeroGrad();
    }

    double train_loss = train_loss_meter.value()(0).scalar<double>();

    // Evaluate on the dev set.
    double val_loss, val_error;
    std::tie(val_loss, val_error) = eval_loop(model, valset);

    std::cout << "Epoch " << e << std::setprecision(3)
              << ": Avg Train Loss: " << train_loss
              << " Validation Loss: " << val_loss
              << " Validation Error (%): " << val_error << std::endl;
  }

Evaluation
----------

The evaluation loop looks a lot like training except without the weight
updates. We just have to be sure to set the model into ``eval`` mode so that
things like Dropout are turned off. Also when we put a model in ``eval`` mode
temporary state needed for the backward pass is not recorded, so this will
require much less memory.

::

  std::pair<double, double> eval_loop(Sequential& model, BatchDataset& dataset) {
    AverageValueMeter loss_meter;
    FrameErrorMeter error_meter;

    // Place the model in eval mode.
    model.eval();
    for (auto& example : dataset) {
      auto inputs = noGrad(example[INPUT_IDX]);
      auto output = model(inputs);

      // Get the predictions in max_ids
      array max_vals, max_ids;
      max(max_vals, max_ids, output.array(), 0);

      auto target = noGrad(example[TARGET_IDX]);

      // Compute and record the prediction error.
      error_meter.add(reorder(max_ids, 1, 0), target.array());

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      loss_meter.add(loss.array().scalar<float>());
    }
    // Place the model back into train mode.
    model.train();

    double error = error_meter.value().scalar<double>();
    double loss = loss_meter.value()(0).scalar<double>();
    return std::make_pair(loss, error);
  }

And then we can compute and report the test error

::

  array test_x;
  array test_y;
  std::tie(test_x, test_y) = load_dataset(data_dir, true);

  td = {{"input", test_x}, {"target", test_y}};
  BatchDataset testset(
    std::make_shared<TensorDataset>(std::vector<af::array>{test_x, test_y}),
    batch_size);

  double test_loss, test_error;
  std::tie(test_loss, test_error) = eval_loop(model, testset);
  std::cout << "Test Loss: " << test_loss << " Test Error (%): " << test_error
            << std::endl;


Running the Example
-------------------

To run the example, build ``Mnist.cpp`` (which is automatically built with flashlight by default), then run

::
   ./Mnist [path to dataset]

After training we should see an output close to

    Test Loss: 0.0373 Test Error (%): 1.1


Conclusion
----------

Here is the complete `source code <todolinktomnist.com>`_ for loading the MNIST
dataset, training and evaluating the model. We've used a number of helpful
flashlight libraries which make this code simple including the ``Sequential``
container, the ``BatchDataset`` and ``DatasetIterator`` classes, and some
derived ``Meter`` classes to keep track of useful stats.

