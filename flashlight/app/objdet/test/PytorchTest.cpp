#include "flashlight/ext/image/fl/models/Resnet34Backbone.h"
#include "flashlight/fl/nn/modules/Conv2D.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
//#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/autograd/Variable.h"

#include <gtest/gtest.h>

fl::Variable param(af::array x) {
  return fl::Variable(x, true);
}

fl::Variable input(af::array x) {
  return fl::Variable(x, false);
}

af::array biasReshape(af::array x) {
  return af::moddims(x, { 1, 1, x.dims(0), 1});
}

bool allClose(
    const af::array& a,
    const af::array& b,
    const double precision = 1e-2) {
  if ((a.numdims() != b.numdims()) || (a.dims() != b.dims())) {
    std::cout << "Shape mismatch " << std::endl;
    return false;
  }
  return (af::max<double>(af::abs(a - b)) < precision);
}

using namespace fl::ext::image;

//TEST(Pytorch, basic_conv) {
  //af::info();
  //const auto pad = fl::PaddingMode::SAME;

  //std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/basic_conv.array";
    
  //af::array x = af::readArray(filename.c_str(), "input");
  //af::array expOutput = af::readArray(filename.c_str(), "output");

  //auto convWeight = param(af::readArray(filename.c_str(), "weight"));
  //auto convBias = param(biasReshape(af::readArray(filename.c_str(), "bias")));

  //fl::Conv2D conv = fl::Conv2D(convWeight, convBias, 1, 1, 1, 1, 1, 1, 1);

  //auto output = conv(input(x));

  //ASSERT_TRUE(allClose(output.array(), expOutput));
//}

//TEST(Pytorch, basic_linear) {
  //af::info();
  //const auto pad = fl::PaddingMode::SAME;

  //std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/basic_linear.array";
    
  //af::array x = af::readArray(filename.c_str(), "input");
  //af::array expOutput = af::readArray(filename.c_str(), "output");

  //auto weight = param(af::readArray(filename.c_str(), "weight"));
  //af_print(weight.array());
  //std::cout << " DIms " << weight.dims() << std::endl;
  ////auto bias = param(af::readArray(filename.c_str(), "bias"));

  //fl::Linear linear = fl::Linear(weight);

  //auto output = linear(input(x));
  //af_print(output.array());
  //af_print(expOutput);

  //ASSERT_TRUE(allClose(output.array(), expOutput));
//}
//
void getBns(
    std::shared_ptr<fl::Module> module, 
    std::vector<std::shared_ptr<fl::Module>>& bns) {
  if(dynamic_cast<fl::BatchNorm*>(module.get())) {
    bns.push_back(module);
  } else if(dynamic_cast<fl::Container*>(module.get())) {
    for(auto mod : dynamic_cast<fl::Container*>(module.get())->modules()) {
      getBns(mod, bns);
    }
  }
};

TEST(Pytorch, resnet34) {
  af::info();
  const auto pad = fl::PaddingMode::SAME;

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/resnet34.array";
    
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  auto resnet34 = fl::ext::image::resnet34();

  int paramSize = resnet34->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 1);
    ASSERT_TRUE(resnet34->param(i).dims() == array.dims());
    resnet34->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(resnet34, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::BatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }
  std::cout << "Number of batchnorms" << bns.size() << std::endl;



  std::cout << resnet34->params()[paramSize - 1].dims() << std::endl;
  std::cout << resnet34->params()[paramSize - 2].dims() << std::endl;
  std::cout << af::readArray(filename.c_str(), paramSize).dims() << std::endl;
  std::cout << af::readArray(filename.c_str(), paramSize - 1).dims() << std::endl;
  resnet34->eval();

  auto output = resnet34->forward(input(x));
  std::cout << output.dims() << std::endl;

  std::cout << output.array().scalar<float>() << std::endl;
  std::cout << expOutput.scalar<float>() << std::endl;

  ASSERT_TRUE(allClose(output.array(), expOutput));
}
