Example: MNIST
==============

In this example, we demonstrate how to implement a simple convolutional network and train it on the `MNIST <http://yann.lecun.com/exdb/mnist/>`_ dataset. Download and unpack the dataset before running.

The full source for the below example can be found in ``examples/Mnist.cpp``.

Data
----

First, we load the data using ``TensorDataset`` and ``BatchDataset``:

::

  Tensor train_x;
  Tensor train_y;
  std::tie(train_x, train_y) = load_dataset(data_dir);

  // Hold out a dev set
  auto val_x = train_x(span, span, seq(0, VAL_SIZE - 1));
  train_x = train_x(span, span, seq(VAL_SIZE, TRAIN_SIZE - 1));
  auto val_y = train_y(seq(0, VAL_SIZE - 1));
  train_y = train_y(seq(VAL_SIZE, TRAIN_SIZE - 1));

  // Make the training batch dataset
  BatchDataset trainset(
      std::make_shared<TensorDataset>(std::vector<Tensor>{train_x, train_y}),
      batch_size);

  // Make the validation batch dataset
  BatchDataset valset(
      std::make_shared<TensorDataset>(std::vector<Tensor>{val_x, val_y}),
      batch_size);


Model
-----

Now, we construct the model:

::

  Sequential model;
  auto pad = PaddingMode::SAME;
  model.add(View(Shape({IM_DIM, IM_DIM, 1, -1})));
  model.add(Conv2D(
      1 /* input channels */,
      32 /* output channels */,
      5 /* kernel width */,
      5 /* kernel height */,
      1 /* stride x */,
      1 /* stride y */,
      pad /* padding mode */,
      pad /* padding mode */));
  model.add(ReLU());
  model.add(Pool2D(
      2 /* kernel width */,
      2 /* kernel height */,
      2 /* stride x */,
      2 /* stride y */));
  model.add(Conv2D(32, 64, 5, 5, 1, 1, pad, pad));
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

First, create an optimizer and run a training loop for a specified number of iterations (enough to pass over the full dataset).

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
      train_loss_meter.add(loss.tensor().scalar<float>());

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

The evaluation loop is similar to the training loop except that it omits updates to model parameters. When evaluating a model, we use ``eval`` mode on the ``Module`` which disables components that should not run at evaluation time (e.g. dropout), and disables gradient computation to save memory.

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
      Tensor max_vals, max_ids;
      max(max_vals, max_ids, output.tensor(), 0);

      auto target = noGrad(example[TARGET_IDX]);

      // Compute and record the prediction error.
      error_meter.add(reorder(max_ids, 1, 0), target.tensor());

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      loss_meter.add(loss.tensor().scalar<float>());
    }
    // Place the model back into train mode.
    model.train();

    double error = error_meter.value().scalar<double>();
    double loss = loss_meter.value()(0).scalar<double>();
    return std::make_pair(loss, error);
  }

Compute and report the test error:

::

  Tensor test_x;
  Tensor test_y;
  std::tie(test_x, test_y) = load_dataset(data_dir, true);

  td = {{"input", test_x}, {"target", test_y}};
  BatchDataset testset(
    std::make_shared<TensorDataset>(std::vector<Tensor>{test_x, test_y}),
    batch_size);

  double test_loss, test_error;
  std::tie(test_loss, test_error) = eval_loop(model, testset);
  std::cout << "Test Loss: " << test_loss << " Test Error (%): " << test_error
            << std::endl;


Running the Example
-------------------

To run the example, build ``Mnist.cpp`` (which is automatically built with flashlight examples by default), then run

::
   ./Mnist [path to dataset]

After training we should see an output close to

    Test Loss: 0.0373 Test Error (%): 1.1
