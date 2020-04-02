#include "flashlight/contrib/modules/Resnet.h"

#include <math.h>

#include "flashlight/nn/Init.h"

namespace fl {

namespace {

Variable readDump(const std::string name, const int modIdx, const int paramIdx, af::dim4 dims) {
  static std::string pytorchDump = "/private/home/padentomasello/tmp/pytorch_dump/";
  std::stringstream ss;
  ss << pytorchDump << name << modIdx << "-" << paramIdx << ".bin";
  const std::string fp = ss.str();
  const int size = dims.elements();
  std::cout << "Reading from " << fp << std::endl;;
  std::ifstream infile(fp, std::ios::binary);
  if(!infile) {
      throw std::invalid_argument("Could not read from fp" + fp);
  }
  std::vector<float> vec(dims.elements());
  infile.read((char*) vec.data(), dims.elements() * sizeof(float));
  if(!infile) {
    throw std::invalid_argument("Could not read from fp" + fp);
  }
  infile.close();
  auto a = af::array(dims, vec.data());
  if (dims.ndims() > 2) {
    af_print(a(af::span, af::span, 0, 0));
  }
  if(dims.ndims() == 2) {
    af_print(a(af::span, 0))
  }
  return param(a);
};

static int moduleCounter = 0;
static std::string pytorchDump = "/private/home/padentomasello/tmp/pytorch_dump/";

Conv2D conv3x3(const int in_c, const int out_c, const int stride,
    const int groups) {
  const auto pad = PaddingMode::SAME;
  auto conv = Conv2D(in_c, out_c, 3, 3, stride, stride, pad, pad, 1, 1, false, groups);
  conv.setParams(
      readDump("conv", moduleCounter, 0, conv.param(0).dims()),
  0);
  moduleCounter++;
  return conv;
}

Conv2D conv1x1(const int in_c, const int out_c, const int stride,
    const int groups) {
  const auto pad = PaddingMode::SAME;
  auto conv = Conv2D(in_c, out_c, 1, 1, stride, stride, pad, pad, 1, 1, false, groups);
  conv.setParams(readDump("conv", moduleCounter, 0, conv.param(0).dims()), 0);
  moduleCounter++;
  //conv.setParams(
      //kaimingNormal(af::dim4(1, 1, in_c / groups, out_c), fanOut, gain),
  //0);
  return conv;
}

BatchNorm batchNorm(const int channels) {
    auto bn = BatchNorm(2, channels);
    bn.setParams(readDump("bn" , moduleCounter, 0, bn.param(0).dims()), 0);
    bn.setParams(readDump("bn" , moduleCounter, 1, bn.param(1).dims()), 1);
    moduleCounter++;
    return bn;
}

}

ConvBnAct::ConvBnAct() = default;

//ConvBnAct::ConvBnAct(
    //const int in_c,
    //const int out_c,
    //const int kw,
    //const int kh,
    //const int sx,
    //const int sy,
    //bool bn,
    //bool act) {
  //const auto pad = PaddingMode::SAME;
  //const bool bias = !bn;

  //auto conv1 = Conv2D(in_c, out_c, kw, kh, sx, sy, pad, pad, 1, 1, bias);
  //const int fanOut = out_c * kw * kh;
  //const float gain = std::sqrt(2.f);
  //conv1.setParams(kaimingNormal(af::dim4(kw, kh, in_c, out_c), fanOut, gain), 0);
  //add(conv1); if (bn) {
    //auto bn = BatchNorm(2, out_c);
    //bn.setParams(constant(1.0, out_c, af::dtype::f32, true), 0);
    //bn.setParams(constant(0.0, out_c, af::dtype::f32, true), 1);
    //add(bn);
  //}   
  //if (act) {
    //add(std::make_shared<fl::ReLU>());
  //}
//}


BasicBlock::BasicBlock() = default;

BasicBlock::BasicBlock(const int in_c, const int out_c, const int stride) {
  add(std::make_shared<Conv2D>(conv3x3(in_c, out_c, stride, 1)));
  add(std::make_shared<BatchNorm>(batchNorm(out_c)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv3x3(out_c, out_c, 1, 1)));
  add(std::make_shared<BatchNorm>(batchNorm(out_c)));
  add(std::make_shared<ReLU>());
  if (in_c != out_c || stride == 2) {
    Sequential downsample;
    downsample.add(conv1x1(in_c, out_c, stride, 1));
    downsample.add(batchNorm(out_c));
    downsample_ = std::make_shared<Sequential>(downsample);
  }
}

std::vector<fl::Variable> BasicBlock::forward(
    const std::vector<fl::Variable>& inputs) {
  auto c1 = module(0);
  auto bn1 = module(1);
  auto relu1 = module(2);
  auto c2 = module(3);
  auto bn2 = module(4);
  auto relu2 = module(5);
  std::vector<fl::Variable> out;
  out = c1->forward(inputs);
  out = bn1->forward(out);
  out = relu1->forward(out);
  out = c2->forward(out);
  out = bn2->forward(out);

  std::vector<fl::Variable> shortcut;
  if (downsample_) {
    shortcut = downsample_->forward(inputs);
  } else {
    shortcut = inputs;
  }
  return relu2->forward({out[0] + shortcut[0]});
}

std::string BasicBlock::prettyString() const {
  return "2-Layer BasicBlock Conv3x3";
}

Bottleneck::Bottleneck() = default;

Bottleneck::Bottleneck(const int in_c, const int width, const int stride) {
  if (in_c != (width * 4) || stride == 2) {
    //downsample_ = std::make_shared<ConvBnAct>(in_c, width * 4, 1, 1, stride, stride, true, false);
  }
  add(std::make_shared<Conv2D>(conv1x1(in_c, width, 1, 1)));
  add(std::make_shared<BatchNorm>(batchNorm(width)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv3x3(width, width, stride, 1)));
  add(std::make_shared<BatchNorm>(batchNorm(width)));
  add(std::make_shared<ReLU>());
  add(std::make_shared<Conv2D>(conv1x1(width, width * 4, 1, 1)));
  add(std::make_shared<BatchNorm>(batchNorm(width * 4)));
  add(std::make_shared<ReLU>());
}

std::vector<fl::Variable> Bottleneck::forward(
    const std::vector<fl::Variable>& inputs) {
  auto c1 = module(0);
  auto bn1 = module(1);
  auto relu1 = module(2);
  auto c2 = module(3);
  auto bn2 = module(4);
  auto relu2 = module(5);
  auto c3 = module(6);
  auto bn3 = module(7);
  auto relu3 = module(8); 
  std::vector<fl::Variable> out;
  out = c1->forward(inputs);
  out = bn1->forward(out);
  out = relu1->forward(out);
  out = c2->forward(out);
  out = bn2->forward(out);
  out = relu2->forward(out);
  out = c3->forward(out);
  out = bn3->forward(out);

  std::vector<fl::Variable> shortcut;
  if (downsample_) {
    shortcut = downsample_->forward(inputs);
  } else {
    shortcut = inputs;
  }
  return relu3->forward({out[0] + shortcut[0]});
}

std::string Bottleneck::prettyString() const {
  return "2-Layer Bottleneck Conv3x3";
}


template <typename Block>
ResNetStage<Block>::ResNetStage(
    const int in_c,
    const int out_c,
    const int num_blocks,
    const int stride) {
  add(Block(in_c, out_c, stride));
  const int expand_channels = out_c * Block::expansion();
  for (int i = 1; i < num_blocks; i++) {
    add(Block(expand_channels, out_c));
  }
}

Sequential resnet50() {
  Sequential model;
  // conv1 -> 244x244x3 -> 112x112x64
  //model.add(ConvBnAct(3, 64, 7, 7, 2, 2));
  // maxpool -> 112x122x64 -> 56x56x64
  model.add(Pool2D(3, 3, 2, 2, 1, 1, PoolingMode::MAX));
  //// conv2_x -> 56x56x64 -> 56x56x64
  model.add(ResNetStage<Bottleneck>(64, 64, 3, 1));
  //// conv3_x -> 56x56x64 -> 28x28x128
  model.add(ResNetStage<Bottleneck>(64 * 4, 128, 4, 2));
  //// conv4_x -> 28x28x128 -> 14x14x256
  model.add(ResNetStage<Bottleneck>(128 * 4, 256, 6, 2));
  //// conv5_x -> 14x14x256 -> 7x7x256
  model.add(ResNetStage<Bottleneck>(256 * 4, 512, 3, 2));
  //// pool 7x7x64 ->
  model.add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  model.add(View({512 * 4, -1, 1, 0}));
  auto linear = Linear(512 * 4, 1000, true);
  const float alpha = 5.0f;
  const float linearGain = std::sqrt(2.0 / (1 + alpha));
  linear.setParams(
      kaimingNormal(af::dim4(1000, 512*4), 512*4, linearGain), 0);
  model.add(linear);
  model.add(View({1000, -1}));
  model.add(LogSoftmax());
  return model;
}

Sequential resnet34() {
  Sequential model;


  // conv1 -> 244x244x3 -> 112x112x64
  auto conv1 = Conv2D(3, 64, 7, 7, 2, 2, 3, 3, 1, 1, false, 1);
  auto param = conv1.param(0);
  conv1.setParams(readDump("conv", moduleCounter, 0, param.dims()) , 0);
  moduleCounter++;
  model.add(conv1);
  model.add(batchNorm(64));
  model.add(ReLU());
  model.add(Pool2D(3, 3, 2, 2, 1, 1, PoolingMode::MAX));

  model.add(ResNetStage<BasicBlock>(64, 64, 3, 1));

  model.add(ResNetStage<BasicBlock>(64, 128, 4, 2));
  // conv4_x -> 28x28x128 -> 14x14x256
  model.add(ResNetStage<BasicBlock>(128, 256, 6, 2));
  // conv5_x -> 14x14x256 -> 7x7x256
  model.add(ResNetStage<BasicBlock>(256, 512, 3, 2));

  model.add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  //model.add(Pool2D(56, 56, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  model.add(View({512, -1, 1, 0}));
  auto linear = Linear(512, 1000, true);
  linear.setParams(readDump("fc", moduleCounter, 0, linear.param(0).dims()), 0);
  linear.setParams(readDump("fc", moduleCounter, 1, linear.param(1).dims()), 1);
  //linear.setParams(readDump("/private/home/padentomasello/tmp/pytorch_dump/fc" + str(moduleCounter) + "1.bin", linear.param(1).dims()), 1);
  //linear.setParams(readDump("/private/home/padentomasello/tmp/pytorch_dump/linear-1.bin", linear.param(1).dims()), 1);
  //const float alpha = 5.0f;
  //const float linearGain = std::sqrt(2.0 / (1 + alpha));
  //linear.setParams( //kaimingNormal(af::dim4(1000, 64), 64, linearGain), 0);
  model.add(linear);
  model.add(View({1000, -1}));
  model.add(LogSoftmax());
  int i = 0;
  for(auto module : model.modules()) {
    std::cout << module->prettyString() << std::endl;
    std::cout << ++i << std::endl;
  }
  return model;
};

/*
Sequential resnet34() {
  Sequential model;
  // conv1 -> 244x244x3 -> 112x112x64
  model.add(ConvBnAct(3, 64, 7, 7, 2, 2));
  
  model.add(Pool2D(3, 3, 2, 2, 1, 1, PoolingMode::MAX));
  // conv2_x -> 56x56x64 -> 56x56x64
  model.add(ResNetStage<BasicBlock>(64, 64, 3, 1));
  // conv3_x -> 56x56x64 -> 28x28x128
  model.add(ResNetStage<BasicBlock>(64, 128, 4, 2));
  // conv4_x -> 28x28x128 -> 14x14x256
  model.add(ResNetStage<BasicBlock>(128, 256, 6, 2));
  // conv5_x -> 14x14x256 -> 7x7x256
  model.add(ResNetStage<BasicBlock>(256, 512, 3, 2));
  // pool 7x7x64 ->
  model.add(Pool2D(7, 7, 1, 1, 0, 0, fl::PoolingMode::AVG_EXCLUDE_PADDING));
  model.add(View({512, -1, 1, 0}));
  auto linear = Linear(512, 1000, true);
  const float alpha = 5.0f;
  const float linearGain = std::sqrt(2.0 / (1 + alpha));
  linear.setParams(
      kaimingNormal(af::dim4(1000, 512), 512, linearGain), 0);
  model.add(linear);
  model.add(View({1000, -1}));
  model.add(LogSoftmax());
  return model;
};
*/

}; // namespace fl
