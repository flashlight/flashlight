#include "vision/dataset/Coco.h"
#include "vision/dataset/Utils.h"

#include <arrayfire.h>

#include <algorithm>
#include <map>

#include "iostream"

namespace {

using namespace fl;
using namespace fl::cv::dataset;

static const int kElementsPerBbox = 5;
static const int kMaxNumLabels = 64;

using BBoxVector = std::vector<float>;
using BBoxLoader = Loader<BBoxVector>;

std::vector<std::string> parseImageFilepaths(const std::string& list_file) {
  std::ifstream ifs(list_file);
  if(!ifs) {
    throw std::runtime_error("Could not open list file: " + list_file);
  }
  std::vector<std::string> filepaths;
  std::string line;
  const std::string delim = "\t";
  while(std::getline(ifs, line)) {
      std::string filepath = line.substr(0, line.find(delim));
      filepaths.emplace_back(filepath);
  }
  return filepaths;
}

std::vector<BBoxVector> parseBoundingBoxes(const std::string& list_file) {
  std::ifstream ifs(list_file);
  if(!ifs) {
    throw std::runtime_error("Could not open list file: " + list_file);
  }
  std::vector<std::vector<float>> labels;
  std::string line;
  const std::string label_delim = "\t";
  const std::string bbox_delim = " ";
  while(std::getline(ifs, line)) {
      std::vector<float> bboxes;
      int pos = line.find(label_delim);
      int next = line.find(bbox_delim, pos + 1);
      while(next != std::string::npos) {
        bboxes.emplace_back(std::stof(line.substr(pos, next - pos)));
        pos = next;
        next = line.find(bbox_delim, pos + 2);
      }
      bboxes.emplace_back(std::stof(line.substr(pos, next - pos)));
      labels.emplace_back(bboxes);
  }
  return labels;
}

BBoxLoader bboxLoader(std::vector<BBoxVector> bboxes) {
  return BBoxLoader(bboxes, [](BBoxVector bbox) {
      const int num_elements = bbox.size();
      // Bounding box coordinates + class label
      const int num_bboxes = num_elements / kElementsPerBbox;
      //assert(num_bboxes < kMaxNumLabels);
      //af::array output = af::constant(-1, kElementsPerBbox, kMaxNumLabels);
      //af::array output = af::constant(-1, kElementsPerBbox, kMaxNumLabels);
      //auto arr = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      //output(af::span, af::seq(num_bboxes)) = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      //return output;
      af::array full = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      af::array result = full(af::seq(0, 3), af::span);
      return result;
  });
}

BBoxLoader classLoader(std::vector<BBoxVector> bboxes) {
  return BBoxLoader(bboxes, [](BBoxVector bbox) {
      const int num_elements = bbox.size();
      // Bounding box coordinates + class label
      const int num_bboxes = num_elements / kElementsPerBbox;
      //assert(num_bboxes < kMaxNumLabels);
      //af::array output = af::constant(-1, kElementsPerBbox, kMaxNumLabels);
      //af::array output = af::constant(-1, kElementsPerBbox, kMaxNumLabels);
      //auto arr = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      //output(af::span, af::seq(num_bboxes)) = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      //return output;
      af::array full = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      af::array result = full(4, af::span);
      return result;
  });
}

af::array batchBboxes(const std::vector<af::array>& in) {
    int total_boxes = 0;
    for(auto arr : in) {
      total_boxes += arr.dims(1);
    }
    af::array output = af::constant(-1, af::dim4(kElementsPerBbox, total_boxes));
    af_print(output);
    int start = 0;
    for(auto arr: in) {
      output(af::span, af::seq(start, start + arr.dims(1) - 1)) = arr(af::span, af::span);
      start += arr.dims(1);
    }
    af_print(output);
    return output;
}

class NamedDataset : public fl::Dataset {

public:
  NamedDataset(
      const std::map<std::string, std::shared_ptr<const Dataset>>& datasets)
    : datasets_(datasets) {
    size_ = 0;
    for (auto it : datasets_) {
      std::string name = it.first;
      std::shared_ptr<const Dataset> dataset = it.second;
      size_ = std::max(dataset->size(), size_);
    }
    std::cout << " size " << size_ << std::endl;
  }

  ~NamedDataset() {}

  std::map<std::string, std::vector<af::array>> getMap(const int64_t idx) const {
    std::map<std::string, std::vector<af::array>> result;
    for (auto it : datasets_) {
      std::string name = it.first;
      std::shared_ptr<const Dataset> dataset = it.second;
      if (idx < dataset->size()) {
        auto f = dataset->get(idx);
        result[name] = f;
      }
    }
    return result;
  }

  int64_t size() const override {
    return size_;
  }

  std::vector<af::array> get(const int64_t idx) const override {
    checkIndexBounds(idx);

    std::map<std::string, std::vector<af::array>> resultMap = getMap(idx);
    std::vector<af::array> result;
    for (auto it : resultMap) {
      std::string name = it.first;
      std::vector<af::array> f = it.second;
      result.insert(
          result.end(),
          std::make_move_iterator(f.begin()),
          std::make_move_iterator(f.end()));
    }
    return result;
  }


  std::vector<af::array> get(const int64_t idx, const std::string& name) const {
    checkIndexBounds(idx);

    std::map<std::string, std::vector<af::array>> result = getMap(idx);
    return result[name];
  }
private:
  const std::map<std::string, std::shared_ptr<const Dataset>>& datasets_;
  int64_t size_;
};

using BatchMergeFunction = std::function<
std::vector<af::array>
(const std::vector<std::vector<af::array>>&)>;

class BatchMergeDataset : public fl::Dataset {

public:
  BatchMergeDataset(std::shared_ptr<const Dataset> dataset,
      int64_t batchsize,
      BatchDatasetPolicy policy /* = BatchDatasetPolicy::INCLUDE_LAST */,
    const BatchMergeFunction& batchfn) 
    : dataset_(dataset),
      batchSize_(batchsize),
      batchPolicy_(policy),
      batchFn_(batchfn) {
    if (!dataset_) {
      throw std::invalid_argument("dataset to be batched is null");
    }
    if (batchSize_ <= 0) {
      throw std::invalid_argument("invalid batch size");
    }
    preBatchSize_ = dataset_->size();
    switch (batchPolicy_) {
      case BatchDatasetPolicy::INCLUDE_LAST:
        size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      case BatchDatasetPolicy::SKIP_LAST:
        size_ = floor(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      case BatchDatasetPolicy::DIVISIBLE_ONLY:
        if (size_ % batchSize_ != 0) {
          throw std::invalid_argument(
              "dataset is not evenly divisible into batches");
        }
        size_ = ceil(static_cast<double>(preBatchSize_) / batchSize_);
        break;
      default:
        throw std::invalid_argument("unknown BatchDatasetPolicy");
    }
  }

  ~BatchMergeDataset() {};

  std::vector<af::array> get(const int64_t idx) const override {
    checkIndexBounds(idx);
    std::vector<std::vector<af::array>> buffer;

    int64_t start = batchSize_ * idx;
    int64_t end = std::min(start + batchSize_, preBatchSize_);
    std::cout << " start " << start << " end " << end << std::endl;

    for (int64_t batchidx = start; batchidx < end; ++batchidx) {
      auto fds = dataset_->get(batchidx);
      std::cout << "fds " << fds.size() << std::endl;
      buffer.emplace_back(fds);
    }
    return batchFn_(buffer);
  }

  int64_t size() const override {
    return size_;
  }

private:
  std::shared_ptr<const Dataset> dataset_;
  int64_t batchSize_;
  BatchDatasetPolicy batchPolicy_;
  BatchMergeFunction batchFn_;

  int64_t preBatchSize_; // Size of the dataset before batching
  int64_t size_;

};

using TransformAllFunction = std::function<
std::vector<af::array>
(const std::vector<af::array>&)>;

class TransformAllDataset : public fl::Dataset {
public:
  TransformAllDataset(std::shared_ptr<const Dataset> dataset,
      TransformAllFunction func) : dataset_(dataset), func_(func) {
  }

  std::vector<af::array> get(const int64_t idx) const override {
    return func_(dataset_->get(idx));
  }

  int64_t size() const override {
    return dataset_->size();
  }

private:
  const TransformAllFunction func_;
  std::shared_ptr<const Dataset> dataset_;
};

af::array makeBatch(
    const std::vector<af::array>& data
    ) {
  // Using default batching function
  if (data.empty()) {
    return af::array();
  }
  auto dims = data[0].dims();

  for (const auto& d : data) {
    if (d.dims() != dims) {
      throw std::invalid_argument("dimension mismatch while batching dataset");
    }
  }

  int ndims = (data[0].elements() > 1) ? dims.ndims() : 0;

  if (ndims >= 4) {
    throw std::invalid_argument("# of dims must be < 4 for batching");
  }
  dims[ndims] = data.size();
  auto batcharr = af::array(dims, data[0].type());

  for (size_t i = 0; i < data.size(); ++i) {
    std::array<af::seq, 4> sel{af::span, af::span, af::span, af::span};
    sel[ndims] = af::seq(i, i);
    batcharr(sel[0], sel[1], sel[2], sel[3]) = data[i];
  }
  return batcharr;
}

}


namespace fl {
namespace cv {
namespace dataset {


CocoData cocoBatchFunc(const std::vector<std::vector<af::array>>& batches) {
  std::vector<af::array> images(batches.size());
  std::transform(batches.begin(), batches.end(), images.begin(),
      [](const std::vector<af::array>& in) { return in[0]; }
  );
  af::array img = makeBatch(images);
  std::vector<af::array> target_bboxes(batches.size());
  std::transform(batches.begin(), batches.end(), target_bboxes.begin(),
      [](const std::vector<af::array>& in) { return in[1]; }
  );
  std::vector<af::array> target_classes(batches.size());
  std::transform(batches.begin(), batches.end(), target_classes.begin(),
      [](const std::vector<af::array>& in) { return in[2]; }
  );
  return { img, target_bboxes, target_classes };
}


CocoDataset::CocoDataset(const std::string& list_file,
      std::vector<ImageTransform>& transformfns
      ) {
    auto images = getImages(list_file, transformfns);
    auto labels = getLabels(list_file);
    auto merged = std::make_shared<MergeDataset>(MergeDataset({images, labels}));
    batched_ = std::make_shared<BatchTransformDataset<CocoData>>(
        merged, 2, BatchDatasetPolicy::INCLUDE_LAST, cocoBatchFunc);

  }

void CocoDataset::resample() {
}

std::shared_ptr<Dataset> CocoDataset::getImages(
    const std::string list_file, 
    std::vector<ImageTransform>& transformfns) {
  const std::vector<std::string> filepaths = parseImageFilepaths(list_file);
  auto images = std::make_shared<FilepathLoader>(jpegLoader(filepaths));
  std::vector<ImageTransform> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);
  return transformed;
}

std::shared_ptr<Dataset> CocoDataset::getLabels(std::string list_file) {
    const std::vector<BBoxVector> bboxes = parseBoundingBoxes(list_file);
    auto bboxLabels = std::make_shared<BBoxLoader>(bboxLoader(bboxes));
    auto classLabels = std::make_shared<BBoxLoader>(classLoader(bboxes));
    return std::make_shared<MergeDataset>(MergeDataset({bboxLabels, classLabels}));
}



int64_t CocoDataset::size() const {
  return batched_->size();
}

CocoData CocoDataset::get(const uint64_t idx) {
  return batched_->get(idx);
}



std::shared_ptr<CocoDataset> coco(
    const std::string& list_file,
    std::vector<ImageTransform>& transformfns)  {
  return std::make_shared<CocoDataset>(list_file, transformfns);

  //const std::vector<std::string> filepaths = parseImageFilepaths(list_file);

  //auto images = std::make_shared<FilepathLoader>(jpegLoader(filepaths));
  //std::vector<ImageTransform> transforms = { compose(transformfns) };
  //auto transformed = std::make_shared<TransformDataset>(images, transforms);

  //const std::vector<BBoxVector> bboxes = parseBoundingBoxes(list_file);
  //auto labels = std::make_shared<BBoxLoader>(bboxLoader(bboxes));


  //auto merged = std::make_shared<MergeDataset>(MergeDataset({transformed, labels, labels}));

  //auto batched = std::make_shared<BatchTransformDataset<CocoData>>(merged, 2, BatchDatasetPolicy::INCLUDE_LAST, cocoBatchFunc)
  ////
  //return batched;
}

} // namespace dataset
} // namespace cv
} // namespace flashlight
