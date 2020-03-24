/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fcntl.h>
#include <sys/stat.h>
#include <memory>
#include <string>

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "flashlight/autograd/autograd.h"
#include "flashlight/nn/nn.h"

using namespace fl;

namespace {

std::string getTmpPath(const std::string& key) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  return std::string("/tmp/test_") + userstr + key + std::string(".mdl");
}

class ContainerTestClass : public Sequential {
 public:
  ContainerTestClass() {}

  void addParam(const Variable& param) {
    params_.push_back(param);
  }

 private:
  FL_SAVE_LOAD_WITH_BASE(Sequential)
};

auto filesizebytes = []() {
  struct stat fileStat;
  auto fd = open(getTmpPath("FileSize").c_str(), O_RDONLY);
  fstat(fd, &fileStat);
  return static_cast<int64_t>(fileStat.st_size);
};

auto paramsizebytes = [](const std::vector<Variable>& parameters) {
  int64_t paramsize = 0;
  for (const auto& param : parameters) {
    paramsize += (param.elements() * af::getSizeOf(param.type()));
  }
  return paramsize;
};

const float kThreshold = 1.01; // within 1%

} // namespace

CEREAL_REGISTER_TYPE(ContainerTestClass)

TEST(NNSerializationTest, Variable) {
  auto testimpl = [](const af::array& arr, bool calc_grad) {
    Variable a(arr, calc_grad);
    std::stringstream ss;
    {
      cereal::BinaryOutputArchive ar(ss);
      ar(a);
    }
    Variable b;
    {
      cereal::BinaryInputArchive ar(ss);
      ar(b);
    }
    ASSERT_TRUE(allClose(a, b));
  };

  testimpl(af::array(), true);
  testimpl(af::randn(3, 6, 7, 8), false);
  testimpl(af::randu(1, 2, 3, 5, af::dtype::b8), false);
  testimpl(af::randu(1, 2, 3, 5, af::dtype::s16), true);
  testimpl(af::randn(5, 6, 7, 9, af::dtype::f64), false);
  testimpl(af::randu(1, 9, 9, 2, af::dtype::s32), true);
  testimpl(af::randu(2, 9, 1, 8, af::dtype::s64), false);
  testimpl(af::randu(100, 150, af::dtype::u8), true);
  testimpl(af::randu(32, 32, 3, af::dtype::u16), false);
  testimpl(af::randn(5, 8, 2, 4, af::dtype::c32), false);
  testimpl(af::randu(2, 3, 4, 9, af::dtype::c64), true);
}

TEST(NNSerializationTest, Linear) {
  auto wt = param(af::randu(4, 3));
  auto bs = param(af::randu(4));
  auto in = input(af::randu(3, 2));
  auto lin = std::make_shared<Linear>(wt, bs);

  save(getTmpPath("Linear"), lin);

  std::shared_ptr<Linear> lin2;
  load(getTmpPath("Linear"), lin2);
  ASSERT_TRUE(lin2);

  ASSERT_TRUE(allParamsClose(*lin2, *lin));
  ASSERT_TRUE(allClose(lin2->forward(in), lin->forward(in)));
}

TEST(NNSerializationTest, Conv2D) {
  auto wt = param(af::randu(5, 5, 2, 4));
  auto bs = param(af::randu(1, 1, 4, 1));
  auto in = input(af::randu(25, 25, 2, 5));
  auto conv = std::make_shared<Conv2D>(wt, bs);

  save(getTmpPath("Conv2D"), conv);

  std::shared_ptr<Conv2D> conv2;
  load(getTmpPath("Conv2D"), conv2);
  ASSERT_TRUE(conv2);

  ASSERT_TRUE(allParamsClose(*conv2, *conv));
  ASSERT_TRUE(allClose(conv2->forward(in), conv->forward(in)));
}

TEST(NNSerializationTest, Pool2D) {
  auto in = input(af::randu(8, 8));
  auto pool = std::make_shared<Pool2D>(2, 3, 1, 1, 1, 1, PoolingMode::MAX);

  save(getTmpPath("Pool2D"), pool);

  std::shared_ptr<Pool2D> pool2;
  load(getTmpPath("Pool2D"), pool2);
  ASSERT_TRUE(pool2);

  ASSERT_TRUE(allParamsClose(*pool2, *pool));
  ASSERT_TRUE(allClose(pool2->forward(in), pool->forward(in)));
}

TEST(NNSerializationTest, BaseModule) {
  auto in = input(af::randu(8, 8));
  ModulePtr dout = std::make_shared<Dropout>(0.75);

  save(getTmpPath("BaseModule"), dout);

  ModulePtr dout2;
  load(getTmpPath("BaseModule"), dout2);
  ASSERT_TRUE(dout2);

  ASSERT_TRUE(allParamsClose(*dout2, *dout));
}

TEST(NNSerializationTest, WeightNormLinear) {
  auto in = input(af::randn(2, 10, 1, 1));
  auto wlin = std::make_shared<WeightNorm>(Linear(2, 3), 0);

  save(getTmpPath("WeightNormLinear"), wlin);

  std::shared_ptr<WeightNorm> wlin2;
  load(getTmpPath("WeightNormLinear"), wlin2);
  ASSERT_TRUE(wlin2);

  ASSERT_TRUE(allParamsClose(*wlin2, *wlin));
  ASSERT_TRUE(
      allClose(wlin2->forward({in}).front(), wlin->forward({in}).front()));
}

TEST(NNSerializationTest, WeightNormConvSeq) {
  auto in = input(af::randn(70, 70, 30, 2));
  auto seq = std::make_shared<Sequential>();
  seq->add(std::make_shared<WeightNorm>(Conv2D(30, 80, 3, 3), 3));
  seq->add(std::make_shared<GatedLinearUnit>(2));
  seq->add(std::make_shared<WeightNorm>(Conv2D(40, 90, 3, 3), 3));
  seq->add(std::make_shared<GatedLinearUnit>(2));
  seq->add(std::make_shared<WeightNorm>(Conv2D(45, 100, 3, 3), 3));
  seq->add(std::make_shared<GatedLinearUnit>(2));
}

TEST(NNSerializationTest, AdaptiveSoftMaxLoss) {
  auto in = input(af::randu(5, 10));
  std::vector<int> h_target{1, 1, 1, 2, 2, 2, 0, 0, 0, 0};
  af::array g_target(10, h_target.data());
  auto target = input(g_target);

  std::vector<int> cutoff{{1, 3}};
  auto activation = std::make_shared<AdaptiveSoftMax>(5, cutoff);
  auto asml = std::make_shared<AdaptiveSoftMaxLoss>(activation);

  save(getTmpPath("AdaptiveSoftMaxLoss"), asml);

  std::shared_ptr<AdaptiveSoftMaxLoss> asml2;
  load(getTmpPath("AdaptiveSoftMaxLoss"), asml2);
  ASSERT_TRUE(asml2);

  ASSERT_TRUE(allParamsClose(*asml2, *asml));
  auto activation2 = asml2->getActivation();
  ASSERT_TRUE(allParamsClose(*activation2, *activation));
  ASSERT_TRUE(allClose(activation2->forward(in), activation->forward(in)));
  ASSERT_TRUE(allClose(asml2->forward(in, target), asml->forward(in, target)));

  auto activation3 = std::make_shared<AdaptiveSoftMax>(5, cutoff);
  auto asml3 = std::make_shared<AdaptiveSoftMaxLoss>(activation3);
  int index = 0;
  for (auto param : asml->params()) {
    asml3->setParams(param, index);
    index++;
  }
  ASSERT_TRUE(allParamsClose(*asml3->getActivation(), *activation));
}

TEST(NNSerializationTest, PrettyString) {
  Sequential seq;
  seq.add(Conv2D(3, 64, 5, 5));
  seq.add(Pool2D(3, 3, 2, 2, 1, 1));
  seq.add(ReLU());
  seq.add(Dropout(0.4));
  seq.add(Linear(5, 10, false));
  seq.add(Tanh());
  seq.add(LeakyReLU(0.2));

  auto prettystr = seq.prettyString();

  std::string expectedstr =
      "Sequential [input -> (0) -> (1) -> (2) -> (3) "
      "-> (4) -> (5) -> (6) -> output]"
      "(0): Conv2D (3->64, 5x5, 1, 1, 0, 0, 1, 1) (with bias)"
      "(1): Pool2D-max (3x3, 2,2, 1,1)"
      "(2): ReLU"
      "(3): Dropout (0.400000)"
      "(4): Linear (5->10) (without bias)"
      "(5): Tanh"
      "(6): LeakyReLU (0.200000)";

  auto remove_ws = [](std::string& str) {
    str.erase(remove_if(str.begin(), str.end(), isspace), str.end());
  };

  remove_ws(expectedstr);
  remove_ws(prettystr);

  ASSERT_EQ(expectedstr, prettystr);
}

TEST(NNSerializationTest, LeNet) {
  auto leNet = std::make_shared<Sequential>();

  leNet->add(Conv2D(3, 6, 5, 5));
  leNet->add(ReLU());
  leNet->add(Pool2D(2, 2, 2, 2));

  leNet->add(Conv2D(6, 16, 5, 5));
  leNet->add(ReLU());
  leNet->add(Pool2D(2, 2, 2, 2));

  leNet->add(View(af::dim4(16 * 5 * 5)));

  leNet->add(Linear(16 * 5 * 5, 120));
  leNet->add(ReLU());

  leNet->add(Linear(120, 84));
  leNet->add(ReLU());

  leNet->add(Linear(84, 10));

  save(getTmpPath("LeNet"), leNet);

  std::shared_ptr<Sequential> leNet2;
  load(getTmpPath("LeNet"), leNet2);
  ASSERT_TRUE(leNet2);

  ASSERT_TRUE(allParamsClose(*leNet2, *leNet));

  auto in = input(af::randu(32, 32, 3));
  ASSERT_TRUE(allClose(leNet2->forward(in), leNet->forward(in)));
}

// Make sure serialized file size if not too high
TEST(NNSerializationTest, FileSize) {
  auto conv = std::make_shared<Conv2D>(300, 600, 10, 10);
  save(getTmpPath("FileSize"), conv);
  ASSERT_LT(filesizebytes(), paramsizebytes(conv->params()) * kThreshold);

  auto seq = Sequential();
  seq.add(Conv2D(64, 64, 3, 100));
  seq.add(ReLU());
  seq.add(Pool2D(2, 2, 2, 2));
  seq.add(Conv2D(64, 64, 100, 200));
  seq.add(ReLU());
  seq.add(Pool2D(2, 2, 2, 2));
  seq.add(Linear(200, 500));
  seq.add(MeanSquaredError());
  save(getTmpPath("FileSize"), seq);
  ASSERT_LT(filesizebytes(), paramsizebytes(seq.params()) * kThreshold);
}

TEST(NNSerializationTest, VariableTwice) {
  Variable v(af::array(1000, 1000), false);
  auto v2 = v; // The array for this variable shouldn't be saved again

  save(getTmpPath("ContainerWithParams"), v2, v);

  struct stat fileStat;
  auto fd = open(getTmpPath("ContainerWithParams").c_str(), O_RDONLY);
  fstat(fd, &fileStat);
  ASSERT_LT(
      static_cast<int64_t>(fileStat.st_size), paramsizebytes({v}) * kThreshold);
}

TEST(NNSerializationTest, ContainerBackward) {
  auto seq = std::make_shared<Sequential>();
  seq->add(Linear(10, 20));
  seq->add(ReLU());
  seq->add(Linear(20, 30));
  save(getTmpPath("ContainerBackward"), static_cast<ModulePtr>(seq));

  ModulePtr seq2;
  load(getTmpPath("ContainerBackward"), seq2);

  auto in = input(af::randu(10, 10));
  auto output = seq2->forward({in}).front();
  output.backward();
  for (auto& p : seq2->params()) {
    ASSERT_TRUE(p.isGradAvailable());
  }
}

TEST(NNSerializationTest, ContainerWithParams) {
  auto seq = std::make_shared<ContainerTestClass>();
  seq->addParam(Variable(af::randu(5, 5), true));
  seq->add(WeightNorm(Linear(10, 20), 0));
  seq->addParam(Variable(af::randu(5, 5), true));
  seq->add(ReLU());
  seq->add(Linear(20, 30));
  seq->addParam(Variable(af::randu(5, 5), true));

  save(getTmpPath("ContainerWithParams"), static_cast<ModulePtr>(seq));

  ModulePtr seq2;
  load(getTmpPath("ContainerWithParams"), seq2);
  ASSERT_TRUE(seq2);

  ASSERT_TRUE(allParamsClose(*seq, *seq2));

  auto in = input(af::randu(10, 10));
  ASSERT_TRUE(allClose(seq->forward(in), seq2->forward({in}).front()));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
