/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Approximate re implementation of the char rnn PyTorch tutorial:
   https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html
   Dataset: https://download.pytorch.org/tutorial/data.zip
*/

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "flashlight/fl/common/Filesystem.h"
#include "flashlight/fl/dataset/datasets.h"
#include "flashlight/fl/meter/meters.h"
#include "flashlight/fl/nn/nn.h"
#include "flashlight/fl/optim/optim.h"
#include "flashlight/fl/tensor/Index.h"
#include "flashlight/fl/tensor/Init.h"

using namespace fl;

// return a random int between [mini, maxi]
int randi(int mini, int maxi) {
  if (maxi < mini) {
    std::swap(maxi, mini);
  }
  return rand() % (maxi - mini + 1) + mini;
}

class ClassificationDataset : public Dataset {
 public:
  using names = std::vector<std::string>;
  std::map<std::string, names> datasets;
  static std::map<std::string, unsigned> Label2Id;
  static std::map<unsigned, std::string> Id2Label;
  unsigned totalExamples = 0;

  // read the folder/lang.txt file and register the examples in the datasets map
  void read(const fs::path& folder, const std::string& lang) {
    auto fp = folder / (lang + ".txt");
    std::cout << "Opening " << fp << std::endl;
    std::ifstream file(fp);
    if (!file.is_open()) {
      throw std::runtime_error("Can't open the input dataset file");
    }
    unsigned id = Label2Id.size();
    Label2Id[lang] = id;
    Id2Label[id] = lang;
    names v;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) {
        continue;
      }
      v.push_back(line);
    }
    totalExamples += v.size();
    std::cout << "Found " << v.size() << " examples for category " << lang
              << ". Total: " << totalExamples << std::endl;
    datasets[lang] = v;
  }

  // Turn a string into an AF array <1 x line_length> of char indices
  static Tensor lineToTensor(const std::string& line) {
    std::vector<float> d;
    for (char c : line) {
      d.push_back((float)(c)); // direct cast of char to float
    }
    return Tensor::fromBuffer(
        {1, static_cast<Dim>(d.size())}, d.data(), MemoryLocation::Host);
  }

  explicit ClassificationDataset(const fs::path& datasetPath) {
    // As found in the dataset folder:
    std::vector<std::string> lang = {
        "Arabic",
        "Greek",
        "Chinese",
        "Czech",
        "Dutch",
        "Japanese",
        "Korean",
        "Russian",
        "English",
        "Scottish",
        "Vietnamese",
        "German",
        "Spanish",
        "French",
        "Polish",
        "Italian",
        "Irish"};
    for (auto& l : lang) {
      read(datasetPath, l);
    }
    for (auto& it : Id2Label) {
      std::cout << it.first << ":" << it.second << ", ";
    }
    std::cout << std::endl;
  }

  // each epoch to go over some percent of the training dataset
  int64_t size() const override {
    return .3f * totalExamples;
  }

  // get a random example: name, category
  std::pair<std::string, std::string> getRandomExample() const {
    std::pair<std::string, std::string> p;
    auto it = datasets.begin();
    unsigned cat = randi(0, datasets.size() - 1); // random category index
    std::advance(it, cat);
    p.second = it->first; // category name
    const auto& v = it->second;
    unsigned nn = v.size();
    unsigned ri = randi(0, nn - 1);
    p.first = v[ri]; // name
    return p;
  }

  // get a (random) example and return a vector of 2 tensors : the input and the
  // expected category index
  std::vector<Tensor> get(const int64_t) const override {
    auto p = getRandomExample();
    const std::string& n = p.first;
    Tensor input = lineToTensor(n);
    std::vector<float> cv;
    cv.push_back(Label2Id[p.second]);
    auto expected = Tensor::fromBuffer({1}, cv.data(), MemoryLocation::Host);
    return {input, expected};
  }
};
std::map<std::string, unsigned> ClassificationDataset::Label2Id;
std::map<unsigned, std::string> ClassificationDataset::Id2Label;

class RnnClassifier : public Container {
 public:
  explicit RnnClassifier(
      unsigned numClasses,
      unsigned vocabSize,
      int hiddenSize = 256,
      unsigned numLayers = 2)
      : embed_(std::make_shared<Embedding>(hiddenSize, vocabSize)),
        rnn_(std::make_shared<RNN>(
            hiddenSize,
            hiddenSize,
            numLayers,
            RnnMode::GRU,
            0 /* Dropout */)),
        linear_(std::make_shared<Linear>(hiddenSize, numClasses)),
        logsoftmax_(0) {
    std::cout << "Creating a RNN Classifier with vocab size: " << vocabSize
              << " and num classes: " << numClasses << std::endl;
    createLayers();
  }

  RnnClassifier(const RnnClassifier& other) {
    copy(other);
    createLayers();
  }

  // The compiler default generated is made explicit for reference.
  // Users must be careful to include move and move assignment
  // constructors where appropriate.
  RnnClassifier(RnnClassifier&& other) = default;

  RnnClassifier& operator=(const RnnClassifier& other) {
    clear();
    copy(other);
    createLayers();
    return *this;
  }

  // The compiler default generated is made explicit for reference.
  // Users must be careful to include move and move assignment
  // constructors where appropriate.
  RnnClassifier& operator=(RnnClassifier&& other) = default;

  void copy(const RnnClassifier& other) {
    embed_ = std::make_shared<Embedding>(*other.embed_);
    rnn_ = std::make_shared<RNN>(*other.rnn_);
    linear_ = std::make_shared<Linear>(*other.linear_);
    logsoftmax_ = other.logsoftmax_;
  }

  void createLayers() {
    add(embed_);
    add(rnn_);
    add(linear_);
  }

  std::unique_ptr<Module> clone() const override {
    return std::make_unique<RnnClassifier>(*this);
  }

  std::vector<Variable> forward(const std::vector<Variable>& inputs) override {
    throw std::runtime_error("Not implemented");
  }

  std::tuple<Variable, Variable, Variable>
  forward(const Variable& input, const Variable& h, const Variable& c) {
    const unsigned numChars = input.dim(1);
    Variable ho, co; // hidden and carry output
    // output should be hs x bs x ts : [ 𝑋𝑖𝑛, 𝑁, 𝑇 ]
    Variable output = embed_->forward(input);
    // The input to the RNN is expected to be of shape [ 𝑋𝑖𝑛, 𝑁, 𝑇 ] where 𝑋𝑖𝑛
    // is the hidden size, 𝑁 is the batch size and 𝑇 is the sequence length.
    std::tie(output, ho, co) = rnn_->forward(output, h, c);
    // The output of the RNN should be of shape [ 𝑋𝑜𝑢𝑡, 𝑁, 𝑇], with 𝑋𝑜𝑢𝑡 to be
    // the hidden_size (unidirectional RNN)
    // Truncate BPTT
    ho.setCalcGrad(false);
    co.setCalcGrad(false);
    output = linear_->forward(output);
    output = logsoftmax_(output);
    output = output(fl::span, fl::span, input.tensor().dim(1) - 1);
    return std::make_tuple(output, ho, co);
  }

  std::string prettyString() const override {
    return "RnnClassifer";
  }

  // Inference on the given input: returns the category index
  unsigned
  infer(const std::string& inputString, const Variable& h, const Variable& c) {
    Tensor ia = ClassificationDataset::lineToTensor(inputString);
    Variable output, ho, co;
    std::tie(output, ho, co) = forward(noGrad(ia), h, c);
    Tensor maxValue, prediction;
    fl::max(maxValue, prediction, output.tensor(), 0);
    unsigned classId = prediction.scalar<unsigned>();
    return classId;
  }

  // Predict the category of the given input, compared with the expected label
  // and print the result
  bool unittest(const std::string& input, const std::string& expectedLabel) {
    Variable output, h, c;
    auto p = ClassificationDataset::Id2Label[infer(input, h, c)];
    const bool passes = p == expectedLabel;
    const std::string s = (passes ? "✓ " : "✗ ");
    std::cout << "input: " << std::setw(20) << input
              << "\t expected: " << expectedLabel << "\t prediction: " << p
              << "\t" << s << std::endl;
    return passes;
  }

 private:
  std::shared_ptr<Embedding> embed_;
  std::shared_ptr<RNN> rnn_;
  std::shared_ptr<Linear> linear_;
  LogSoftmax logsoftmax_;
};

int main(int argc, char** argv) {
  fl::init();
  std::cout << "RnnClassification (path to the data dir) (learning rate) (num "
               "epochs) (hiddensize)"
            << std::endl;
  std::cout << "Dataset : https://download.pytorch.org/tutorial/data.zip"
            << std::endl;
  if (argc < 2) {
    std::cout << "To setup the dataset: " << std::endl;
    std::cout << "wget https://download.pytorch.org/tutorial/data.zip"
              << std::endl;
    std::cout << "unzip data.zip" << std::endl;
    std::cout << "./RnnClassification data/names" << std::endl;
    return 0;
  }
  fs::path dataDir = argv[1];

  // To reproduce the pytorch tutorial, the dataset is not split in
  // train/dev/test sub datasets but random samples are simply picked from the
  // overall dataset:
  ClassificationDataset trainSet(dataDir);

  const float learningRate = argc > 2 ? std::stof(argv[2]) : 0.1;
  const int epochs = argc > 3 ? std::stol(argv[3]) : 6;
  const unsigned hiddenSize = argc > 4 ? std::stol(argv[4]) : 256;
  const float momentum = 0.9;
  const float maxGradNorm = 0.25;

  RnnClassifier model(
      ClassificationDataset::Label2Id.size(),
      256, // input vocab size set to 256 to support any possible character,
           // ascii or not
      hiddenSize);
  // https://fl.readthedocs.io/en/latest/modules.html#categoricalcrossentropy
  CategoricalCrossEntropy criterion;
  auto opt = SGDOptimizer(model.params(), learningRate, momentum);

  // Each epoch to go over a small percent of the dataset
  for (int e = 0; e < epochs; e++) {
    AverageValueMeter trainLossMeter;
    Variable output, h, c;
    const int kInputIdx = 0, kTargetIdx = 1;
    for (auto& example : trainSet) {
      std::tie(output, h, c) = model.forward(noGrad(example[kInputIdx]), h, c);
      auto target = noGrad(example[kTargetIdx]);
      // Computes the categorical cross entropy loss:
      // The input is expected to contain log-probabilities for each class.
      // The targets should be the index of the ground truth class for each
      // input example.
      auto loss = criterion(output, target);
      trainLossMeter.add(loss.tensor().scalar<float>(), target.elements());
      opt.zeroGrad();
      loss.backward();
      // Clipping is a must have to avoid exploding gradients:
      clipGradNorm(model.params(), maxGradNorm);
      opt.step();
    }

    double trainLoss = trainLossMeter.value()[0];
    std::cout << "Epoch " << e + 1 << std::setprecision(3)
              << " - Train Loss: " << trainLoss << std::endl;

    // compute the accuracy confusion matrix:
    const unsigned nCategories = ClassificationDataset::Label2Id.size();
    Tensor confusion = fl::full({nCategories, nCategories}, 0.);
    // Go through a bunch of examples and record which are correctly guessed
    float numMatch = 0, nConfusion = 1000;
    for (unsigned i = 0; i < nConfusion; ++i) {
      auto p = trainSet.getRandomExample();
      unsigned pred = model.infer(p.first, h, c);
      unsigned correctPred = ClassificationDataset::Label2Id[p.second];
      if (pred == correctPred) {
        ++numMatch;
      }
      confusion(correctPred, pred) = confusion(correctPred, pred) + 1;
    }
    confusion = confusion /
        fl::tile(fl::sum(confusion, {1}), {1, nCategories}); // average
    std::cout << "Global accuracy=" << numMatch / nConfusion << "\t ";
    for (unsigned i = 0; i < nCategories; ++i) {
      std::cout << ClassificationDataset::Id2Label[i] << ":" << std::fixed
                << std::setprecision(2) << confusion(i, i).scalar<float>()
                << " ";
    }
    std::cout << std::endl;
  }
  // List of names not in the training dataset
  const std::vector<std::pair<std::string, std::string>> quickList = {
      {"Samad", "Arabic"},
      {"Papademos", "Greek"},
      {"Birovsky", "Czech"},
      {"Wai", "Chinese"},
      {"Nikolaev", "Russian"},
      {"Washington", "English"},
      {"Voltaire", "French"},
      {"Pfeiffer", "German"},
      {"Tambellini", "Italian"}};
  for (auto& p : quickList) {
    model.unittest(p.first, p.second);
  }

  while (true) {
    std::string name;
    std::cout << "Enter a surname and press enter to classify it: ";
    std::cin >> name;
    Variable output, h, c;
    std::cout << ClassificationDataset::Id2Label[model.infer(name, h, c)]
              << " ?" << std::endl;
  }
  std::cout << "Finished" << std::endl;
  return 0;
}
