/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * A flashlight introduction to Convolutional Neural Networks on MNIST. The
 * model is based on this tutorial from TensorFlow:
 * https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py
 *
 * To run this demo first download and unpack the mnist dataset from:
 * http://yann.lecun.com/exdb/mnist/
 *
 * Once downloaded run the program with:
 * ./Mnist <path_to_data>
 *
 * Final output should be close to:
 *   Test Loss: 0.0373 Test Error (%): 1.1
 */

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include <arrayfire.h>
#include "flashlight/fl/flashlight.h"

using namespace af;
using namespace fl;

namespace {
const int TRAIN_SIZE = 60000;
const int VAL_SIZE = 5000; /* Held-out from train. */
const int TEST_SIZE = 10000;
const int IM_DIM = 28;
const int PIXEL_MAX = 255;
const int INPUT_IDX = 0;
const int TARGET_IDX = 1;

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

  double error = error_meter.value();
  double loss = loss_meter.value()[0];
  return std::make_pair(loss, error);
}

std::pair<array, array> load_dataset(
    const std::string& data_dir,
    bool test = false);

} // namespace

int main(int argc, char** argv) {
  fl::init();
  if (argc != 2) {
    throw af::exception("You must pass a data directory.");
  }
  fl::setSeed(1);
  std::string data_dir = argv[1];

  float learning_rate = 1e-2;
  int epochs = 10;
  int batch_size = 64;

  array train_x;
  array train_y;
  std::tie(train_x, train_y) = load_dataset(data_dir);

  // Hold out a dev set
  auto val_x = train_x(span, span, 0, seq(0, VAL_SIZE - 1));
  train_x = train_x(span, span, 0, seq(VAL_SIZE, TRAIN_SIZE - 1));
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

  Sequential model;
  auto pad = PaddingMode::SAME;
  model.add(View(af::dim4(IM_DIM, IM_DIM, 1, -1)));
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
  model.add(View(af::dim4(7 * 7 * 64, -1)));
  model.add(Linear(7 * 7 * 64, 1024));
  model.add(ReLU());
  model.add(Dropout(0.5));
  model.add(Linear(1024, 10));
  model.add(LogSoftmax());

  // Make the optimizer
  SGDOptimizer opt(model.params(), learning_rate);

  // The main training loop
  for (int e = 0; e < epochs; e++) {
    AverageValueMeter train_loss_meter;

    // Get an iterator over the data
    for (auto& example : trainset) {
      // Make a Variable from the input array.
      auto inputs = noGrad(example[INPUT_IDX]);

      // Get the activations from the model.
      auto output = model(inputs);

      // Make a Variable from the target array.
      auto target = noGrad(example[TARGET_IDX]);

      // Compute and record the loss.
      auto loss = categoricalCrossEntropy(output, target);
      train_loss_meter.add(loss.array().scalar<float>());

      // Backprop, update the weights and then zero the gradients.
      loss.backward();
      opt.step();
      opt.zeroGrad();
    }

    double train_loss = train_loss_meter.value()[0];

    // Evaluate on the dev set.
    double val_loss, val_error;
    std::tie(val_loss, val_error) = eval_loop(model, valset);

    std::cout << "Epoch " << e << std::setprecision(3)
              << ": Avg Train Loss: " << train_loss
              << " Validation Loss: " << val_loss
              << " Validation Error (%): " << val_error << std::endl;
  }

  array test_x;
  array test_y;
  std::tie(test_x, test_y) = load_dataset(data_dir, true);

  BatchDataset testset(
      std::make_shared<TensorDataset>(std::vector<af::array>{test_x, test_y}),
      batch_size);

  double test_loss, test_error;
  std::tie(test_loss, test_error) = eval_loop(model, testset);
  std::cout << "Test Loss: " << test_loss << " Test Error (%): " << test_error
            << std::endl;

  return 0;
}

namespace {

// MNIST Data loading functions below.

int read_int(std::ifstream& f) {
  int d = 0;
  int c;
  for (int i = 0; i < sizeof(int); i++) {
    c = 0;
    f.read((char*)&c, 1);
    d |= (c << (8 * (sizeof(int) - i - 1)));
  }
  return d;
}

template <typename T>
array load_data(
    const std::string& im_file,
    const std::vector<long long int>& dims) {
  std::ifstream file(im_file, std::ios::binary);
  if (!file.is_open()) {
    throw af::exception("[mnist:load_data] Can't find MNIST file.");
  }
  read_int(file); // unused magic
  size_t elems = 1;
  for (auto d : dims) {
    int read_d = read_int(file);
    elems *= read_d;
    if (read_d != d) {
      throw af::exception("[mnist:load_data] Unexpected MNIST dimension.");
    }
  }

  std::vector<T> data;
  data.reserve(elems);
  for (int i = 0; i < elems; i++) {
    unsigned char tmp;
    file.read((char*)&tmp, sizeof(tmp));
    data.push_back(tmp);
  }

  std::vector<long long int> rdims(dims.rbegin(), dims.rend());
  // af is column-major
  dim4 af_dims(rdims.size(), rdims.data());
  return array(af_dims, data.data());
}

std::pair<array, array> load_dataset(
    const std::string& data_dir,
    bool test /* = false */) {
  std::string f = test ? "t10k" : "train";
  int size = test ? TEST_SIZE : TRAIN_SIZE;

  std::string image_file = data_dir + "/" + f + "-images-idx3-ubyte";
  array ims = load_data<float>(image_file, {size, IM_DIM, IM_DIM});
  ims = moddims(ims, {IM_DIM, IM_DIM, 1, size});
  // Rescale to [-0.5,  0.5]
  ims = (ims - PIXEL_MAX / 2) / PIXEL_MAX;

  std::string label_file = data_dir + "/" + f + "-labels-idx1-ubyte";
  array labels = load_data<int>(label_file, {size});

  return std::make_pair(ims, labels);
}
} // namespace
