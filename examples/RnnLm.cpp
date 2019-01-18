/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * A Flashlight introduction to RNNs.
 * The model is based on this RNN LM tutorial from TensorFlow
 * https://git.io/fp9oy
 *
 * The data can be found here:
 * http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
 *
 * To run the example, build ``RnnLm.cpp`` (which is automatically built with
 * flashlight by default), then run:
 *
 *    ./RnnLm [path to dataset]
 *
 * The final output should be close to:
 *   Test Loss: 4.75 Test Perplexity: 115
 */
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "flashlight/flashlight.h"

using namespace fl;

namespace {

class Preprocessor {
 public:
  explicit Preprocessor(std::string dataset_path);

  int to_int(std::string word) {
    return word_to_int[word];
  }

  int vocab_size() {
    return word_to_int.size();
  }

  static const std::string eos;

 private:
  std::unordered_map<std::string, int> word_to_int;
};

class LMDataset : public Dataset {
 public:
  LMDataset(
      std::string dataset_path,
      int batch_size,
      int time_steps,
      Preprocessor& preproc);

  int64_t size() const override {
    return (data.dims(1) - 1) / time_steps;
  }

  std::vector<af::array> get(const int64_t idx) const override;

 private:
  int time_steps;
  af::array data;
};

class RnnLm : public Container {
 public:
  explicit RnnLm(int vocab_size, int hidden_size = 200)
      : embed(Embedding(hidden_size, vocab_size)),
        rnn(
            RNN(hidden_size,
                hidden_size,
                2, /* Num layers. */
                RnnMode::LSTM,
                0 /* Dropout */)),
        linear(Linear(hidden_size, vocab_size)) {
    add(embed);
    add(rnn);
    add(linear);
    add(LogSoftmax());
  }

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override {
    auto inSz = inputs.size();
    if (inSz < 1 || inSz > 3) {
      throw std::invalid_argument("Invalid inputs size");
    }
    return rnn(inputs);
  }

  std::tuple<Variable, Variable, Variable>
  forward(const Variable& input, const Variable& h, const Variable& c) {
    auto output = embed(input);
    Variable ho, co;
    std::tie(output, ho, co) = rnn(output, h, c);

    // Truncate BPTT
    ho.setCalcGrad(false);
    co.setCalcGrad(false);

    output = linear(output);
    return std::make_tuple(output, ho, co);
  }

  std::tuple<Variable, Variable, Variable>
  operator()(const Variable& input, const Variable& h, const Variable& c) {
    return forward(input, h, c);
  }

  std::string prettyString() const override {
    return "RnnLm";
  }

 private:
  Embedding embed;
  RNN rnn;
  Linear linear;
};

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    throw af::exception("You must pass a data directory.");
  }
  std::string data_dir = argv[1];

  std::string train_dir = data_dir + "/ptb.train.txt";
  std::string valid_dir = data_dir + "/ptb.valid.txt";
  std::string test_dir = data_dir + "/ptb.test.txt";

  // Since we also average the loss by time_steps
  float learning_rate = 20;
  float max_grad_norm = 0.25;

  int epochs = 10;
  int anneal_after_epoch = 4;
  int batch_size = 20;
  int time_steps = 20;

  Preprocessor preproc(train_dir);
  LMDataset trainset(train_dir, batch_size, time_steps, preproc);
  LMDataset valset(valid_dir, batch_size, time_steps, preproc);
  const int kInputIdx = 0, kTargetIdx = 1;

  int vocab_size = preproc.vocab_size();
  std::cout << "Vocab size: " << vocab_size << std::endl;

  RnnLm model(vocab_size);
  CategoricalCrossEntropy criterion;

  SGDOptimizer opt(model.params(), learning_rate);

  auto eval_loop = [&model, &criterion](LMDataset& dataset) {
    AverageValueMeter avg_loss_meter;
    Variable output, h, c;
    for (auto& example : dataset) {
      std::tie(output, h, c) = model(noGrad(example[kInputIdx]), h, c);
      auto target = noGrad(example[kTargetIdx]);
      auto loss = criterion(output, target);
      avg_loss_meter.add(loss.array().scalar<float>(), target.elements());
    }
    return avg_loss_meter.value()[0];
  };

  for (int e = 0; e < epochs; e++) {
    AverageValueMeter train_loss_meter;
    TimeMeter timer(true);
    timer.resume();

    Variable output, h, c;

    if (e >= anneal_after_epoch) {
      opt.setLr(opt.getLr() / 2);
    }

    for (auto& example : trainset) {
      std::tie(output, h, c) = model(noGrad(example[kInputIdx]), h, c);

      auto target = noGrad(example[kTargetIdx]);

      auto loss = criterion(output, target);
      train_loss_meter.add(loss.array().scalar<float>(), target.elements());

      opt.zeroGrad();
      loss.backward();

      clipGradNorm(model.params(), max_grad_norm);
      opt.step();

      af::sync();
      timer.incUnit();
    }

    double train_loss = train_loss_meter.value()[0];
    double val_loss = eval_loop(valset);
    double iter_time = timer.value();

    std::cout << "Epoch " << e + 1 << std::setprecision(3)
              << " - Train Loss: " << train_loss
              << " Validation Loss: " << val_loss
              << " Validation Perplexity: " << std::exp(val_loss)
              << " Time per iteration (ms): " << iter_time * 1000 << std::endl;
  }

  LMDataset testset(test_dir, batch_size, time_steps, preproc);

  double test_loss = eval_loop(testset);
  std::cout << " Test Loss: " << test_loss
            << " Test Perplexity: " << std::exp(test_loss) << std::endl;

  return 0;
}

const std::string Preprocessor::eos = "<eos>";

Preprocessor::Preprocessor(std::string dataset_path) {
  std::ifstream file(dataset_path);
  if (!file.is_open()) {
    throw af::exception("[Preprocessor::Preprocessor] Can't find file.");
  }
  int v = 0;
  std::string word;
  while (file >> word) {
    if (word_to_int.find(word) == word_to_int.end()) {
      word_to_int[word] = v++;
    }
  }
  word_to_int[eos] = v;
}

LMDataset::LMDataset(
    std::string dataset_path,
    int batch_size,
    int time_steps,
    Preprocessor& preproc)
    : time_steps(time_steps) {
  std::vector<int> words;
  std::ifstream file(dataset_path);
  if (!file.is_open()) {
    throw af::exception("[LMDataset::LMDataset] Can't find file.");
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream ss(line);
    std::string word;
    while (ss >> word) {
      words.push_back(preproc.to_int(word));
    }
    words.push_back(preproc.to_int(Preprocessor::eos));
  }

  int words_per_batch = words.size() / batch_size;
  words.resize(batch_size * words_per_batch);

  data = transpose(af::array(words_per_batch, batch_size, words.data()));
}

std::vector<af::array> LMDataset::get(const int64_t idx) const {
  int start = idx * time_steps;
  int end = (idx + 1) * time_steps - 1;
  return {data(af::span, af::seq(start, end)),
          data(af::span, af::seq(start + 1, end + 1))};
}
