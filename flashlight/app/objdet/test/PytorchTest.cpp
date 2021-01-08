#include "flashlight/ext/image/fl/models/Resnet34Backbone.h"
#include "flashlight/fl/nn/modules/Conv2D.h"
#include "flashlight/ext/image/fl/models/Resnet.h"
#include "flashlight/ext/image/fl/models/Resnet50Backbone.h"
#include "flashlight/app/objdet/nn/PositionalEmbeddingSine.h"
#include "flashlight/app/objdet/nn/Transformer.h"
//#include "flashlight/fl/flashlight.h"
#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/app/objdet/criterion/Hungarian.h"
#include "flashlight/app/objdet/dataset/Coco.h"
#include "flashlight/app/objdet/criterion/SetCriterion.h"
#include "flashlight/app/objdet/nn/Detr.h"
#include "flashlight/fl/autograd/autograd.h"
#include "flashlight/fl/common/Defines.h"
#include "flashlight/fl/optim/optim.h"

#include <gtest/gtest.h>

using namespace fl::app::objdet;
//using namespace fl;

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
    std::cout << " A dims " << a.dims() << std::endl;
    std::cout << " B dims " << b.dims() << std::endl;
    std::cout << "Shape mismatch " << std::endl;
    return false;
  }
  std::cout << " Max " << af::max<double>(af::abs(a - b)) << std::endl;
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
  if(dynamic_cast<fl::FrozenBatchNorm*>(module.get())) {
    bns.push_back(module);
  } else if(dynamic_cast<fl::Container*>(module.get())) {
    for(auto mod : dynamic_cast<fl::Container*>(module.get())->modules()) {
      getBns(mod, bns);
    }
  }
};

#if 0

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

  resnet34->eval();

  auto output = resnet34->forward(input(x));
  std::cout << output.dims() << std::endl;

  std::cout << output.array().scalar<float>() << std::endl;
  std::cout << expOutput.scalar<float>() << std::endl;

  ASSERT_TRUE(allClose(output.array(), expOutput));
}

TEST(Pytorch, basic_array) {

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/basic_array.array";
    
  af::array x = af::readArray(filename.c_str(), "input");
}

TEST(Pytorch, basic_output) {

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/basic_output.array";
    
  af::array x = af::readArray(filename.c_str(), "output");
}

TEST(Pytorch, pos_embedd) {
  af::info();
  const auto pad = fl::PaddingMode::SAME;

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/pos_embedding.array";
    
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  auto model = std::make_shared<PositionalEmbeddingSine>(4);
  auto inputDims = x.dims();
  af::dim4 modifiedDims = { inputDims[0], inputDims[1], 1, inputDims[2] };

  auto input = fl::Variable(1 - af::moddims(x, modifiedDims), false);
  auto output = model->forward(input);

  ASSERT_TRUE(allClose(output.array(), expOutput));
}

TEST(Pytorch, matcher_test) {

  const int batchSize = 2;

  std::string bboxFilename = "/private/home/padentomasello/scratch/pytorch_testing/matcher_test0_bboxes.array";
  std::string labelFilename = "/private/home/padentomasello/scratch/pytorch_testing/matcher_test0_labels.array";
  std::string inputFilename = "/private/home/padentomasello/scratch/pytorch_testing/matcher_test0input.array";

  std::vector<fl::Variable> bboxes, labels;
  for(int i = 0; i < batchSize; i++) {
    bboxes.push_back({af::readArray(bboxFilename.c_str(), i), false});
    labels.push_back({af::readArray(labelFilename.c_str(), i), false});
  }

  auto predBoxes = fl::Variable(af::readArray(inputFilename.c_str(), "pred_boxes"), false);
  auto predLogits = fl::Variable(af::readArray(inputFilename.c_str(), "pred_logits"), false);

  auto matcher = HungarianMatcher(1.0f, 1.0f, 1.0f);

  std::vector<std::pair<af::array, af::array>> results = matcher.forward(predBoxes, predLogits, bboxes, labels);
  for(auto result : results) {
    af_print(result.first);
    af_print(result.second);
  }
    
}

#endif

//TEST(Pytorch, weightedCategoricalCrossEntropy) {

  //std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/weight_cross_entropy.array";
  //auto inputArray  = af::readArray(filename.c_str(), "input");
  
  //auto targetArray  = af::readArray(filename.c_str(), "target");
  //auto weightArray  = af::readArray(filename.c_str(), "weight");
  //af_print(inputArray);
  //auto input = fl::Variable(inputArray, true);
  //auto softmaxed = fl::logSoftmax(input, 0);
  //af_print(softmaxed.array());

  //auto loss = weightedCategoricalCrossEntropy(
      //softmaxed,
      //{ targetArray, false },
      //{ weightArray, false},
      //fl::ReduceMode::MEAN,
      //-1
      //);
  //loss.backward();
  //af_print(input.grad().array());


//}

//TEST(Pytorch, weightedCategoricalCrossEntropySum) {

  //std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/weight_cross_entropy.array";
  //auto inputArray  = af::readArray(filename.c_str(), "input");
  
  //auto targetArray  = af::readArray(filename.c_str(), "target");
  //auto weightArray  = af::readArray(filename.c_str(), "weight");
  //auto softmaxed = fl::logSoftmax({ inputArray, false}, 0);

  //auto result = weightedCategoricalCrossEntropy(
      //softmaxed,
      //{ targetArray, false },
      //{ weightArray, false},
      //fl::ReduceMode::SUM,
      //-1
      //);
  //af_print(result.array());


//}

//TEST(Pytorch, categoricalCrossEntropy) {

  //std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/weight_cross_entropy.array";
  //auto inputArray  = af::readArray(filename.c_str(), "input");
  
  //auto targetArray  = af::readArray(filename.c_str(), "target");
  //auto weightArray  = af::readArray(filename.c_str(), "weight");
  //auto softmaxed = fl::logSoftmax({ inputArray, false}, 0);

  //auto result = categoricalCrossEntropy(
      //softmaxed,
      //{ targetArray, false },
      //fl::ReduceMode::MEAN,
      //-1
      //);
  //af_print(result.array());


//}


TEST(Pytorch, val_dataset) {

  std::string filepath = "/private/home/padentomasello/data/coco3/";
  const std::vector<float> mean = {0.485, 0.456, 0.406};
  const std::vector<float> std = {0.229, 0.224, 0.225};
  std::vector<ImageTransform> val_transforms = {
      // Resize shortest side to 256, then take a center crop
      //resizeTransform(256),
      //centerCropTransform(224),
      normalizeImage(mean, std)
  };
  auto valDataset = CocoDataset(
      filepath + "val.lst", 
      val_transforms,
      0,
      1,
      1,
      1,
      1,
      true);

  af_print(valDataset.get(0).images(0, 0, 0));


  ASSERT_TRUE(true);

}

TEST(Pytorch, set_crit) {

  const int batchSize = 2;

  std::string bboxFilename = "/private/home/padentomasello/scratch/pytorch_testing/set_criterion_bboxes.array";
  std::string labelFilename = "/private/home/padentomasello/scratch/pytorch_testing/set_criterion_labels.array";
  std::string inputFilename = "/private/home/padentomasello/scratch/pytorch_testing/set_criterion_input.array";
  std::string lossFilename = "/private/home/padentomasello/scratch/pytorch_testing/set_criterion_loss.array";
  std::string gradFilename = "/private/home/padentomasello/scratch/pytorch_testing/set_criterion_grad.array";

  std::vector<fl::Variable> bboxes, labels;
  for(int i = 0; i < batchSize; i++) {
    bboxes.push_back({af::readArray(bboxFilename.c_str(), i).as(f32), false});
    labels.push_back({af::readArray(labelFilename.c_str(), i).as(f32), false});
  }

  auto predBoxes = fl::Variable(af::readArray(inputFilename.c_str(), "pred_boxes"), true);
  auto predLogits = fl::Variable(af::readArray(inputFilename.c_str(), "pred_logits"), true);

  auto matcher = HungarianMatcher(1.0f, 1.0f, 1.0f);

  std::unordered_map<std::string, float> lossWeightsBase = 
        { { "loss_ce" , 1.f} ,
        { "loss_giou", 1.f },
        { "loss_bbox", 1.f }
  };
  SetCriterion::LossDict losses;

  auto criterion = SetCriterion(
      predLogits.dims(0) - 1,
      matcher,
      lossWeightsBase,
      0.1f,
      losses);

  auto results = criterion.forward(predBoxes, predLogits, bboxes, labels);
  for(auto result : results) {
    //result.second.backward(true);
    std::cout << "Checking: " << result.first << std::endl;
    af_print(result.second.array());
    auto expOutput = af::readArray(lossFilename.c_str(), result.first.c_str());
    af_print(expOutput);
    ASSERT_TRUE(allClose(result.second.array(), af::readArray(lossFilename.c_str(), result.first.c_str())));
  }

  auto sum = results["loss_ce_0"] + results["loss_giou_0"] + results["loss_bbox_0"];
  sum.backward();
  //results["loss_ce_0"].backward(true);
  //results["loss_giou_0"].backward(true);
  //results["loss_bbox_0"].backward(true);
  //
  auto expPredBoxGrad = af::readArray(gradFilename.c_str(), "pred_boxes");
  auto expPredLogitsGrad = af::readArray(gradFilename.c_str(), "pred_logits");
  ASSERT_TRUE(allClose(predBoxes.grad().array(), expPredBoxGrad));
  ASSERT_TRUE(allClose(predLogits.grad().array(), expPredLogitsGrad));
  //af_print(predBoxes.grad().array());
  //af_print(predLogits.grad().array());
}

TEST(Pytorch, detr_backbone) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/detr_backbone.array";
  af::array image = af::readArray(filename.c_str(), "image");
  af::array expOutput = af::readArray(filename.c_str(), "output");


  auto model = std::make_shared<fl::ext::image::Resnet50Backbone>();
  std::vector<fl::Variable> inputs = { 
    fl::Variable(image, false), 
  };


  int paramSize = model->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 2);
    if(i == 264) {
      array = af::moddims(array, { 1, 1, 256, 1});
    }
    ASSERT_TRUE(model->param(i).dims() == array.dims());
    model->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(model, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::FrozenBatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }
  //std::string modelPath = "/checkpoint/padentomasello/models/resnet50/from_pytorch_fbn2";
  //fl::save(modelPath, model);
  //fl::load(modelPath, model);

  auto output = model->forward(inputs)[0];
  //af_print(output.array());
  ASSERT_TRUE(allClose(output.array(), expOutput));
}

TEST(Pytorch, detr_backbone_grad) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/detr_backbone_grad.array";
  af::array image = af::readArray(filename.c_str(), "image");
  af::array expOutput = af::readArray(filename.c_str(), "output");


  auto model = std::make_shared<fl::ext::image::Resnet50Backbone>();
  std::vector<fl::Variable> inputs = { 
    fl::Variable(image, true), 
  };


  int paramSize = model->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 2);
    if(i == 264) {
      array = af::moddims(array, { 1, 1, 256, 1});
    }
    ASSERT_TRUE(model->param(i).dims() == array.dims());
    model->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(model, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::FrozenBatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }
  //std::string modelPath = "/checkpoint/padentomasello/models/resnet50/from_pytorch_fbn2";
  //fl::save(modelPath, model);
  //fl::load(modelPath, model);
  model->train();

  auto output = model->forward(inputs)[0];
  auto loss = sum(output, {0, 1, 2, 3});
  loss.backward();
  af_print(loss.array());
  af_print(inputs[0].grad().array());
  //af_print(output.array());
  ASSERT_TRUE(allClose(output.array(), expOutput));
}

TEST(Pytorch, detr) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/detr.array";
  af::array image = af::readArray(filename.c_str(), "image");
  //af::array queries = af::readArray(filename.c_str(), "queries");
  af::array mask = af::readArray(filename.c_str(), "mask");
  mask = af::moddims(mask, { mask.dims(0), mask.dims(1), 1, mask.dims(2)});
  mask = 1 - mask;
  //af::array pos = af::readArray(filename.c_str(), "pos");
  //af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = 256;
  const int numHead = 8;

  const int numLayers = 6;

  auto transformer = std::make_shared<Transformer>(embeddingDim, numHead, numLayers, numLayers, 2048, 0.1f);
  auto backbone = std::make_shared<fl::ext::image::Resnet50Backbone>();

  auto model = std::make_shared<Detr>(transformer, backbone, 256, 91, 100, true);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(image, true), 
    fl::Variable(mask, true), // mask
  };

  int paramSize = model->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 4);
    if(i == 264) {
      array = af::moddims(array, { 1, 1, 256, 1});
    }
    //std::cout << " i " << i << std::endl;
    //std::cout << " Array " << array.dims() << std::endl;
    //std::cout << " Model " << model->param(i).dims() << std::endl;
    ASSERT_TRUE(model->param(i).dims() == array.dims());
    model->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(backbone, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::FrozenBatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }
  //model->eval();
  //backbone->eval();
  //std::string modelPath = "/checkpoint/padentomasello/models/detr/from_pytorch";
  std::string modelPath = "/checkpoint/padentomasello/models/detr/pytorch_initializaition_dropout";
  fl::save(modelPath, model);
  //fl::load(modelPath, model);

  auto outputs = model->forward(inputs);
  af::array expPredLogits = af::readArray(filename.c_str(), "pred_logits");
  af::array expPredBoxes = af::readArray(filename.c_str(), "pred_boxes");
  auto predLogits = outputs[0];
  auto predBoxes = outputs[1];
  ASSERT_TRUE(allClose(predBoxes.array(), expPredBoxes));
  ASSERT_TRUE(allClose(predLogits.array(), expPredLogits));
  auto matcher = HungarianMatcher(1.0f, 5.0f, 2.0f);

  std::unordered_map<std::string, float> lossWeightsBase = 
        { { "loss_ce" , 1.f} ,
        { "loss_giou", 5.f },
        { "loss_bbox", 2.f }
  };
  SetCriterion::LossDict losses;

  auto criterion = SetCriterion(
      91,
      matcher,
      lossWeightsBase,
      0.5f,
      losses);
  std::vector<fl::Variable> targetClasses = { {af::readArray(filename.c_str(), "target_labels"), false}};
  std::vector<fl::Variable> targetBoxes = { { af::readArray(filename.c_str(), "target_boxes"), false }};
  auto results = criterion.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(auto result : results) {
    std::cout << "Checking: " << result.first << std::endl;
    af_print(result.second.array());
    //ASSERT_TRUE(allClose(result.second.array(), af::readArray(lossFilename.c_str(), result.first.c_str())));
  }
  results["loss_giou_0"].backward();

}

TEST(Pytorch, detr_grad_update) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/detr_grad_update.array";
  af::array image = af::readArray(filename.c_str(), "image");
  //af::array queries = af::readArray(filename.c_str(), "queries");
  af::array mask = af::readArray(filename.c_str(), "mask");
  mask = af::moddims(mask, { mask.dims(0), mask.dims(1), 1, mask.dims(2)});
  mask = 1 - mask;

  const int embeddingDim = 256;
  const int numHead = 8;

  const int numLayers = 6;

  auto transformer = std::make_shared<Transformer>(embeddingDim, numHead, numLayers, numLayers, 2048, 0.0f);
  auto backbone = std::make_shared<fl::ext::image::Resnet50Backbone>();

  auto model = std::make_shared<Detr>(transformer, backbone, 256, 91, 100, true);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(image, true), 
    fl::Variable(mask, true), // mask
  };

  int paramSize = model->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 4);
    if(i == 264) {
      array = af::moddims(array, { 1, 1, 256, 1});
    }
    ASSERT_TRUE(model->param(i).dims() == array.dims());
    model->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(backbone, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::FrozenBatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }
  float lr = 0.0001;
  float wd = 0.0000;
  float max_norm = 0.0;
  const float beta1 = 0.9;
  const float beta2 = 0.999;
  const float epsilon = 1e-8;
  auto opt = std::make_shared<fl::AdamOptimizer>(
      model->paramsWithoutBackbone(), lr, beta1, beta2,  epsilon, wd);
  //auto opt = std::make_shared<fl::SGDOptimizer>(
      //model->params(), lr, 0.9, wd);
  auto opt2 = std::make_shared<fl::AdamOptimizer>(model->backboneParams(), lr * 0.1, beta1, beta2, epsilon, wd);
  model->train();
  opt->zeroGrad();
  opt2->zeroGrad();

  auto outputs = model->forward(inputs);
  af::array expPredLogits = af::readArray(filename.c_str(), "pred_logits");
  af::array expPredBoxes = af::readArray(filename.c_str(), "pred_boxes");
  auto predLogits = outputs[0];
  auto predBoxes = outputs[1];
  ASSERT_TRUE(allClose(predLogits.array()(af::span, af::span, af::span, predLogits.dims(3) - 1), expPredLogits));
  ASSERT_TRUE(allClose(predBoxes.array()(af::span, af::span, af::span, predBoxes.dims(3) - 1), expPredBoxes));
  auto matcher = HungarianMatcher(1.0f, 5.0f, 2.0f);

  std::unordered_map<std::string, float> lossWeightsBase = 
        { { "loss_ce" , 1.f} ,
        { "loss_giou", 2.f },
        { "loss_bbox", 5.f }
  };
  std::unordered_map<std::string, float> lossWeights;
  for(int i = 0; i < 6; i++) {
    for(auto l : lossWeightsBase) {
      std::string key = l.first + "_" + std::to_string(i);
      lossWeights[key] = l.second;
    }
  }
  SetCriterion::LossDict losses;

  auto criterion = SetCriterion(
      91,
      matcher,
      lossWeights,
      0.1f,
      losses);
  std::vector<fl::Variable> targetClasses = { {af::readArray(filename.c_str(), "target_labels"), false}};
  std::vector<fl::Variable> targetBoxes = { { af::readArray(filename.c_str(), "target_boxes"), false }};
  auto loss = criterion.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(auto result : loss) {
    std::cout << "Checking: " << result.first << std::endl;
    //af_print(result.second.array());
    //ASSERT_TRUE(allClose(result.second.array(), af::readArray(lossFilename.c_str(), result.first.c_str())));
  }
  auto accumLoss = fl::Variable(af::constant(0, 1), true);
  auto weightDict = criterion.getWeightDict();
  for(auto losses : loss) {
    fl::Variable scaled_loss = weightDict[losses.first] * losses.second;
    accumLoss = scaled_loss + accumLoss;
  }



  opt->zeroGrad();
  opt2->zeroGrad();
  accumLoss.backward();
  //af_print(accumLoss.array());
  //if (max_norm > 0) {
    //fl::clipGradNorm(model->params(), max_norm);
  //}
  //af_print(inputs[0].grad().array());
  ASSERT_TRUE(allClose(inputs[0].grad().array(), af::readArray(filename.c_str(), "image_grad")));

  //opt->zeroGrad();
  //opt2->zeroGrad();
  //accumLoss.backward();

  opt->step();
  opt2->step();

  //int paramSize = model->params().size();
  std::string filename_postupdate = "/private/home/padentomasello/scratch/pytorch_testing/detr_grad_post_update.array";
  af_print(model->param(0).array());
  std::cout << "here" << std::endl;
  af_print(model->param(0).grad().array());
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename_postupdate.c_str(), i);
    //af_print(array);
    if(i == 264) {
      array = af::moddims(array, { 1, 1, 256, 1});
    }
    //af_print(model->param(i).array());
    //std::cout << " i " << i << std::endl;
    //std::cout << " Array " << array.dims() << std::endl;
    //std::cout << " Model " << model->param(i).dims() << std::endl;
    ASSERT_TRUE(allClose(model->param(i).array(),array));
  }


  //results["loss_giou_0"].backward();

}

TEST(Pytorch, detr_grad) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/detr_grad.array";
  af::array image = af::readArray(filename.c_str(), "image");
  //af::array queries = af::readArray(filename.c_str(), "queries");
  af::array mask = af::readArray(filename.c_str(), "mask");
  mask = af::moddims(mask, { mask.dims(0), mask.dims(1), 1, mask.dims(2)});
  mask = 1 - mask;
  //af::array pos = af::readArray(filename.c_str(), "pos");
  //af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = 256;
  const int numHead = 8;

  const int numLayers = 6;

  auto transformer = std::make_shared<Transformer>(embeddingDim, numHead, numLayers, numLayers, 2048, 0.0f);
  auto backbone = std::make_shared<fl::ext::image::Resnet50Backbone>();

  auto model = std::make_shared<Detr>(transformer, backbone, 256, 91, 100, true);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(image, true), 
    fl::Variable(mask, true), // mask
  };

  int paramSize = model->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 4);
    if(i == 264) {
      array = af::moddims(array, { 1, 1, 256, 1});
    }
    //std::cout << " i " << i << std::endl;
    //std::cout << " Array " << array.dims() << std::endl;
    //std::cout << " Model " << model->param(i).dims() << std::endl;
    ASSERT_TRUE(model->param(i).dims() == array.dims());
    model->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(backbone, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::FrozenBatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }
  //model->eval();
  //backbone->eval();
  //std::string modelPath = "/checkpoint/padentomasello/models/detr/from_pytorch";
  //std::string modelPath = "/checkpoint/padentomasello/models/detr/pytorch_initializaition";
  //fl::save(modelPath, model);
  //fl::load(modelPath, model);

  auto outputs = model->forward(inputs);
  af::array expPredLogits = af::readArray(filename.c_str(), "pred_logits");
  af::array expPredBoxes = af::readArray(filename.c_str(), "pred_boxes");
  auto predLogits = outputs[0];
  auto predBoxes = outputs[1];
  ASSERT_TRUE(allClose(predLogits.array()(af::span, af::span, af::span, predLogits.dims(3) - 1), expPredLogits));
  ASSERT_TRUE(allClose(predBoxes.array()(af::span, af::span, af::span, predBoxes.dims(3) - 1), expPredBoxes));
  auto matcher = HungarianMatcher(1.0f, 5.0f, 2.0f);

  std::unordered_map<std::string, float> lossWeightsBase = 
        { { "loss_ce" , 1.f} ,
        { "loss_giou", 2.f },
        { "loss_bbox", 5.f }
  };
  std::unordered_map<std::string, float> lossWeights;
  for(int i = 0; i < 6; i++) {
    for(auto l : lossWeightsBase) {
      std::string key = l.first + "_" + std::to_string(i);
      lossWeights[key] = l.second;
    }
  }
  SetCriterion::LossDict losses;

  auto criterion = SetCriterion(
      91,
      matcher,
      lossWeights,
      0.1f,
      losses);
  std::vector<fl::Variable> targetClasses = { {af::readArray(filename.c_str(), "target_labels"), false}};
  std::vector<fl::Variable> targetBoxes = { { af::readArray(filename.c_str(), "target_boxes"), false }};
  auto loss = criterion.forward(predBoxes, predLogits, targetBoxes, targetClasses);
  for(auto result : loss) {
    std::cout << "Checking: " << result.first << std::endl;
    af_print(result.second.array());
    //ASSERT_TRUE(allClose(result.second.array(), af::readArray(lossFilename.c_str(), result.first.c_str())));
  }
  auto accumLoss = fl::Variable(af::constant(0, 1), true);
  auto weightDict = criterion.getWeightDict();
  for(auto losses : loss) {
    fl::Variable scaled_loss = weightDict[losses.first] * losses.second;
    //meters[losses.first].add(losses.second.array());
    //meters[losses.first + "_weighted"].add(scaled_loss.array());
    accumLoss = scaled_loss + accumLoss;
  }
  accumLoss.backward();
  af_print(accumLoss.array());
  af_print(inputs[0].grad().array());
  ASSERT_TRUE(allClose(inputs[0].grad().array(), af::readArray(filename.c_str(), "image_grad")));
  //results["loss_giou_0"].backward();

}

#if 0




TEST(Pytorch, multihead_attention) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/multi_headed_attention.array";
  af::array x = af::readArray(filename.c_str(), "input");
  af::array keyPaddingMask = af::readArray(filename.c_str(), "mask");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = x.dims(0);
  const int headDim = x.dims(0);
  const int numHead = 2;

  auto model = MultiheadAttention(embeddingDim, embeddingDim / numHead, numHead);

  int paramSize = model.params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 2);
    std::cout << " i " << i << std::endl;
    std::cout << "model dims " << model.param(i).dims() << std::endl;
    std::cout << " array dims " << array.dims() << std::endl;
    ASSERT_TRUE(model.param(i).dims() == array.dims());
    model.setParams(param(array), i);
  }

  af_print(x);

  auto output = model.forward({x, false}, {x, false}, {x, false}, {1 - keyPaddingMask, false})[0];
  ASSERT_TRUE(allClose(output.array(), expOutput));

}



TEST(Pytorch, transformer_encoder_layer) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/transformer_encoder_layer.array";
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = x.dims(0);
  const int headDim = x.dims(0);
  const int numHead = 1;

  auto model = TransformerEncoderLayer(embeddingDim, headDim, 128, numHead, 0.0);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(x, false), 
    fl::Variable({}, false), // mask
    fl::Variable({}, false)  // pos
  };

  int paramSize = model.params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 1);
    ASSERT_TRUE(model.param(i).dims() == array.dims());
    model.setParams(param(array), i);
  }

  auto output = model.forward(inputs)[0];
  ASSERT_TRUE(allClose(output.array(), expOutput));
}



TEST(Pytorch, transformer_decoder_layer) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/transformer_decoder_layer.array";
  af::array memory = af::readArray(filename.c_str(), "memory");
  af::array queries = af::readArray(filename.c_str(), "queries");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = memory.dims(0);
  const int headDim = memory.dims(0);
  const int numHead = 1;

  auto model = TransformerDecoderLayer(embeddingDim, headDim, 128, numHead, 0.0);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(queries, false), 
    fl::Variable(memory, false), // mask
    fl::Variable({}, false)  // pos
  };

  int paramSize = model.params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 2);
    ASSERT_TRUE(model.param(i).dims() == array.dims());
    model.setParams(param(array), i);
  }

  auto output = model.forward(inputs)[0];
  ASSERT_TRUE(allClose(output.array(), expOutput));
}


TEST(Pytorch, transformer_multilayer_encoder) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/transformer_encoder.array";
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = x.dims(0);
  const int headDim = x.dims(0);
  const int numHead = 1;
  const int numLayers = 6;

  auto model = TransformerEncoder(embeddingDim, headDim, 128, numHead, numLayers, 0.0);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(x, false), 
    fl::Variable({}, false), // mask
    fl::Variable({}, false)  // pos
  };

  int paramSize = model.params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 1);
    std::cout << " i " << i << std::endl;
    std::cout << " Array " << array.dims() << std::endl;
    std::cout << " Model " << model.param(i).dims() << std::endl;
    ASSERT_TRUE(model.param(i).dims() == array.dims());
    model.setParams(param(array), i);
  }

  auto output = model.forward(inputs)[0];
  ASSERT_TRUE(allClose(output.array(), expOutput));
}


//TEST(Pytorch, transformer_multilayer_decoder) {


  //std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/transformer_decoder.array";
  //af::array memory = af::readArray(filename.c_str(), "memory");
  //af::array queries = af::readArray(filename.c_str(), "queries");
  //af::array expOutput = af::readArray(filename.c_str(), "output");

  //const int embeddingDim = memory.dims(0);
  //const int headDim = memory.dims(0);
  //const int numHead = 1;

  //const int numLayers = 2;

  //auto model = TransformerDecoder(embeddingDim, headDim, 128, numHead, numLayers, 0.0);
  //std::vector<fl::Variable> inputs = { 
    //fl::Variable(queries, false), 
    //fl::Variable(memory, false), // mask
    //fl::Variable({}, false)  // pos
  //};

  //int paramSize = model.params().size();
  //for(int i = 0; i < paramSize; i++) {
    //auto array = af::readArray(filename.c_str(), i + 2);
    //std::cout << " i " << i << std::endl;
    //std::cout << " Array " << array.dims() << std::endl;
    //std::cout << " Model " << model.param(i).dims() << std::endl;
    //ASSERT_TRUE(model.param(i).dims() == array.dims());
    //model.setParams(param(array), i);
  //}

  //auto output = model.forward(inputs)[0];
  //ASSERT_TRUE(allClose(output.array(), expOutput));
//}
//
TEST(Pytorch, transformer) {


  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/transformer.array";
  af::array src = af::readArray(filename.c_str(), "src");
  af::array queries = af::readArray(filename.c_str(), "queries");
  af::array mask = af::readArray(filename.c_str(), "mask");
  mask = af::moddims(mask, { mask.dims(0), mask.dims(1), 1, mask.dims(2)});
  mask = 1 - mask;
  af::array pos = af::readArray(filename.c_str(), "pos");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  const int embeddingDim = 8;
  const int numHead = 8;

  const int numLayers = 6;

  auto model = Transformer(embeddingDim, numHead, numLayers, numLayers, 2048, 0.0f);
  std::vector<fl::Variable> inputs = { 
    fl::Variable(src, false), 
    fl::Variable(mask, false), // mask
    fl::Variable(queries, false),  // pos
    fl::Variable(pos, false)  // pos
  };

  int paramSize = model.params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 4);
    std::cout << " i " << i << std::endl;
    std::cout << " Array " << array.dims() << std::endl;
    std::cout << " Model " << model.param(i).dims() << std::endl;
    ASSERT_TRUE(model.param(i).dims() == array.dims());
    model.setParams(param(array), i);
  }

  auto output = model.forward(inputs)[0];
  af_print(output.array());
  ASSERT_TRUE(allClose(output.array(), expOutput));
}








TEST(Pytorch, resnet50) {
  af::info();
  const auto pad = fl::PaddingMode::SAME;

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/resnet50.array";
    
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  auto resnet50 = fl::ext::image::resnet50();

  int paramSize = resnet50->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 1);
    ASSERT_TRUE(resnet50->param(i).dims() == array.dims());
    resnet50->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(resnet50, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::BatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }

  resnet50->eval();

  auto output = resnet50->forward(input(x));
  std::cout << output.dims() << std::endl;
  std::cout << " Max " << af::max<double>(output.array()) << std::endl;
  std::cout << " Max " << af::max<double>(expOutput) << std::endl;

  //std::cout << output.array().scalar<float>() << std::endl;
  //std::cout << expOutput.scalar<float>() << std::endl;

  ASSERT_TRUE(allClose(output.array(), expOutput));
}

TEST(Pytorch, bottleneck) {
  af::info();
  const auto pad = fl::PaddingMode::SAME;

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/bottleneck.array";
    
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  auto model = std::make_shared<fl::ext::image::ResNetBottleneckBlock>(40, 10, 1);

  int paramSize = model->params().size();
  for(int i = 0; i < paramSize; i++) {
    auto array = af::readArray(filename.c_str(), i + 1);
    ASSERT_TRUE(model->param(i).dims() == array.dims());
    model->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(model, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::BatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }

  model->eval();

  auto output = model->forward({ input(x) })[0];

  ASSERT_TRUE(allClose(output.array(), expOutput));
}



TEST(Pytorch, resnet50_backbone) {
  af::info();
  const auto pad = fl::PaddingMode::SAME;

  std::string filename = "/private/home/padentomasello/scratch/pytorch_testing/resnet50_backbone.array";
    
  af::array x = af::readArray(filename.c_str(), "input");
  af::array expOutput = af::readArray(filename.c_str(), "output");

  auto model = std::make_shared<fl::app::objdet::Detr>();

  int paramSize = resnet50->params().size();
  // Hack! Don't read the last two!
  for(int i = 0; i < paramSize - 2; i++) {
    auto array = af::readArray(filename.c_str(), i + 1);
    std::cout << " i " << i << std::endl;
    std::cout << " Array " << array.dims() << std::endl;
    std::cout << " Parma " << resnet50->param(i).dims() << std::endl;
    ASSERT_TRUE(resnet50->param(i).dims() == array.dims());
    resnet50->setParams(param(array), i);
  }

  std::vector<std::shared_ptr<fl::Module>> bns;
  getBns(resnet50, bns);

  int i = 0;
  for(auto bn : bns) {
    auto bn_ptr = dynamic_cast<fl::BatchNorm*>(bn.get());
    bn_ptr->setRunningMean(af::readArray((filename + "running").c_str(), i));
    i++;
    bn_ptr->setRunningVar(af::readArray((filename + "running").c_str(), i));
    i++;
  }

  resnet50->eval();

  auto output = resnet50->forward({ input(x) })[1];
  std::string modelPath = "/checkpoint/padentomasello/models/resnet50/from_pytorch";
  fl::save(modelPath, resnet50);
  std::cout << output.dims() << std::endl;
  std::cout << " Max " << af::max<double>(output.array()) << std::endl;
  std::cout << " Max " << af::max<double>(expOutput) << std::endl;

  //std::cout << output.array().scalar<float>() << std::endl;
  //std::cout << expOutput.scalar<float>() << std::endl;

  ASSERT_TRUE(allClose(output.array(), expOutput));
}

#endif
