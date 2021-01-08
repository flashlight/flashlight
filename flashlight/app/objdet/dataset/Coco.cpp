#include "flashlight/app/objdet/dataset/BoxUtils.h"
#include "flashlight/app/objdet/dataset/Coco.h"
#include "flashlight/ext/image/af/Transforms.h"
#include "flashlight/app/objdet/dataset/Transforms.h"
#include "flashlight/ext/image/af/Jpeg.h"
#include "flashlight/ext/image/fl/dataset/DistributedDataset.h"
#include "flashlight/ext/image/fl/dataset/LoaderDataset.h"

#include <arrayfire.h>

#include <assert.h>
#include <algorithm>
#include <map>

namespace {

using namespace fl::app::objdet;
using namespace fl::ext::image;
using namespace fl;

using BBoxVector = std::vector<float>;
using BBoxLoader = LoaderDataset<BBoxVector>;
using FilepathLoader = LoaderDataset<std::string>;

static const int kElementsPerBbox = 5;
static const int kMaxNumLabels = 64;

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
      if(next == std::string::npos) {
        labels.emplace_back(bboxes);
        continue;
      }
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

std::shared_ptr<const Dataset> bboxLoader(std::vector<BBoxVector> bboxes) {
  return std::make_shared<BBoxLoader>(bboxes, [](BBoxVector bbox) {
      const int num_elements = bbox.size();
      // Bounding box coordinates + class label
      const int num_bboxes = num_elements / kElementsPerBbox;
      if (num_bboxes > 0) {
        af::array full = af::array(kElementsPerBbox, num_bboxes, bbox.data());
        std::vector<af::array> result = { full(af::seq(0, 3), af::span) };
        return result;
      } else {
        std::vector<af::array> result = { af::array(0, 1, 1, 1) };
        return result;
      }
  });
}

//BBoxLoader bboxLoader(std::vector<BBoxVector> bboxes) {
  //return BBoxLoader(bboxes, [](BBoxVector bbox) {
      //const int num_elements = bbox.size();
      //// Bounding box coordinates + class label
      //const int num_bboxes = num_elements / kElementsPerBbox;
      //af::array full = af::array(kElementsPerBbox, num_bboxes, bbox.data());
      //af::array result = full(af::seq(0, 3), af::span);
      //return result;
  //});
//}
//
//

std::pair<af::array, af::array> makeImageAndMaskBatch(
    const std::vector<af::array>& data
    ) {
  // Using default batching function
  if (data.empty()) {
    return std::make_pair(af::array(), af::array());
  }
  //auto dims = data[0].dims();

  int maxW = -1;
  int maxH = -1;;

  for (const auto& d : data) {
    int w = d.dims(0);
    int h = d.dims(1);
    maxW = std::max(w, maxW);
    maxH = std::max(h, maxH);
  }
  //// TODO TESTING!!!!!!!!
  //// TODO 
  //maxW = 801;
  //maxH = 801;
  //// TODO
  af::dim4 dims = { maxW, maxH, 3, static_cast<long>(data.size()) };
  af::dim4 maskDims = { maxW, maxH, 1, static_cast<long>(data.size()) };

  //int ndims = (data[0].elements() > 1) ? dims.ndims() : 0;

  //if (ndims >= 4) {
    //throw std::invalid_argument("# of dims must be < 4 for batching");
  //}
  //dims[ndims] = data.size();
  auto batcharr = af::constant(0, dims);

  //auto maskarr = af::constant(true, dims, b8);
  auto maskarr = af::constant(0, maskDims);

  for (size_t i = 0; i < data.size(); ++i) {
    af::array sample = data[i];
    af::dim4 dims = sample.dims();
    int w = dims[0];
    int h = dims[1];
    batcharr(af::seq(0, w - 1), af::seq(0, h - 1), af::span, af::seq(i, i)) = data[i];
    //maskarr(af::seq(0, w - 1), af::seq(0, h - 1), af::span, af::seq(i, i)) = af::constant(false, dims, b8);
    maskarr(af::seq(0, w - 1), af::seq(0, h - 1), af::span, af::seq(i, i)) = af::constant(1, { w, h });
  }
  return std::make_pair(batcharr, maskarr);
}

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

CocoData cocoBatchFunc(const std::vector<std::vector<af::array>>& batches) {
  // TODO padentomasello refactor
  std::vector<af::array> images(batches.size());
  std::vector<af::array> image_sizes(batches.size());
  std::vector<af::array> image_ids(batches.size());
  std::vector<af::array> target_bboxes(batches.size());
  std::vector<af::array> target_classes(batches.size());

  std::transform(batches.begin(), batches.end(), images.begin(),
      [](const std::vector<af::array>& in) { return in[0]; }
  );
  std::transform(batches.begin(), batches.end(), image_sizes.begin(),
      [](const std::vector<af::array>& in) { return in[1]; }
  );
  std::transform(batches.begin(), batches.end(), image_ids.begin(),
      [](const std::vector<af::array>& in) { return in[2]; }
  );

  std::transform(batches.begin(), batches.end(), target_bboxes.begin(),
      [](const std::vector<af::array>& in) { return in[3]; }
  );
  std::transform(batches.begin(), batches.end(), target_classes.begin(),
      [](const std::vector<af::array>& in) { return in[4]; }
  );

  af::array imageBatch, masks;
  std::tie(imageBatch, masks) = makeImageAndMaskBatch(images);
  return {
    imageBatch,
    masks,
    makeBatch(image_sizes),
    makeBatch(image_ids),
    target_bboxes,
    target_classes
  };
}

std::shared_ptr<Dataset> transform(
    std::shared_ptr<Dataset> in,
    std::vector<ImageTransform>& transforms) {
  std::vector<ImageTransform> composed = { compose(transforms) } ;
  return std::make_shared<TransformDataset>(in, composed);
}
std::shared_ptr<Dataset> merge(const std::vector<std::shared_ptr<const Dataset>>& in) {
  return std::make_shared<MergeDataset>(in);
}

std::shared_ptr<Dataset> classLoader(std::vector<BBoxVector> bboxes) {
  return std::make_shared<BBoxLoader>(bboxes, [](BBoxVector bbox) {
      const int num_elements = bbox.size();
      const int num_bboxes = num_elements / kElementsPerBbox;
      if(num_bboxes > 0) {
        af::array full = af::array(kElementsPerBbox, num_bboxes, bbox.data());
        std::vector<af::array> result = { full(4, af::span) };
        return result;
      } else {
        std::vector<af::array> result = { af::array(0, 1, 1, 1) };
        return result;
      }
  });
}

int64_t getImageId(const std::string fp) {
    const std::string slash("/");
    const std::string period(".");
    int start = fp.rfind(slash);
    int end = fp.rfind(period);
    std::string substring = fp.substr(start + 1, end - start);
    return std::stol(substring);
}

std::shared_ptr<Dataset> cocoDataLoader(std::vector<std::string> fps) {
  return std::make_shared<FilepathLoader>(fps, [](const std::string& fp) {
      af::array image = loadJpeg(fp);
      // TODO (padentomasello) check this against Pytorch eval code
      long long int imageSizeArray[] = { image.dims(1), image.dims(0) };
      af::array targetSize = af::array(2, imageSizeArray);
      af::array imageId = af::constant(getImageId(fp), 1, s64);
      std::vector<af::array> result = { image, targetSize, imageId };
      return result;
  });
}

using TransformAllFunction = std::function<std::vector<af::array>(const std::vector<af::array>&)>;


class TransformAllDataset : public Dataset {

public:

  TransformAllDataset(
    std::shared_ptr<const Dataset> dataset,
    TransformAllFunction fn) : dataset_(dataset), fn_(fn) {};

  std::vector<af::array> get(const int64_t idx) const override {
    return fn_(dataset_->get(idx));
  }

  int64_t size() const override { return dataset_->size(); };

private:
  std::shared_ptr<const Dataset> dataset_;
  const TransformAllFunction fn_;

};

}


namespace fl {
namespace app {
namespace objdet {

// TODO move into common namespace
af::array resizeSmallest(const af::array& in, const int resize) {
    const int w = in.dims(0);
    const int h = in.dims(1);
    int th, tw;
    if (h > w) {
      th = (resize * h) / w;
      tw = resize;
    } else {
      th = resize;
      tw = (resize * w) / h;
    }
    return af::resize(in, tw, th, AF_INTERP_BILINEAR);
}

af::array resize(const af::array& in, const int resize) {
  return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
}

//af::array resize(const af::array& in, const int ow, const int oh) {
  //return af::resize(in, resize, resize, AF_INTERP_BILINEAR);
//}

std::vector<af::array> Normalize(const std::vector<af::array> in) {
  auto boxes = in[3];

  if(!boxes.isempty()) {
    auto image = in[0];
    auto w = float(image.dims(0));
    auto h = float(image.dims(1));

    boxes = xyxy_to_cxcywh(boxes);
    const std::vector<float> ratioVector = { w, h, w, h };
    af::array ratioArray = af::array(4, ratioVector.data());
    boxes = af::batchFunc(boxes, ratioArray, af::operator/);
  }
  return { in[0], in[1], in[2], boxes, in[4] };

}

int randomInt(int min, int max) {
  return std::rand() % (max - min + 1) + min;
}

TransformAllFunction randomSelect(std::vector<TransformAllFunction> fns)
{
  return [fns](const std::vector<af::array>& in) {
    TransformAllFunction randomFunc = fns[std::rand() % fns.size()];
    return randomFunc(in);
  };
};

TransformAllFunction randomSizeCrop(int minSize, int maxSize) {
  return [minSize, maxSize](const std::vector<af::array>& in) {
    const af::array& image = in[0];
    const int w = image.dims(0);
    const int h = image.dims(1);
    const int tw = randomInt(minSize, std::min(w, maxSize));
    const int th = randomInt(minSize, std::min(h, maxSize));
    const int x = std::rand() % (w - tw + 1);
    const int y = std::rand() % (h - th + 1);
    return crop(in, x, y, tw, th);
  };
};

TransformAllFunction randomResize(
    std::vector<int> sizes,
    int maxsize) {
  assert(sizes.size() > 0);

  auto getSize = [](const af::array& in, int size, int maxSize = 0) {
    int w = in.dims(0);
    int h = in.dims(1);
    //long size;
    if(maxSize > 0) {
      float minOriginalSize = std::min(w, h);
      float maxOriginalSize = std::max(w, h);
      if (maxOriginalSize / minOriginalSize * size > maxSize) {
        size = round(maxSize * minOriginalSize / maxOriginalSize);
      }
    }

    if( (w <= h && w == size) || (h <= w && h == size)) {
        return std::make_pair(w, h);
    }
    int ow, oh;
    if ( w < h ) {
      ow = size;
      oh = size * h / w;
    } else {
      oh = size;
      ow = size * w / h;
    }
    return std::make_pair(ow, oh);
  };

  auto resizeCoco = [sizes, maxsize, getSize](std::vector<af::array> in) {
    assert(in.size() == 5);
    assert(sizes.size() > 0);
    int randomIndex = rand() % sizes.size();
    int size = sizes[randomIndex];
    const af::array originalImage = in[0];
    auto output_size = getSize(originalImage, size, maxsize);
    const af::dim4 originalDims = originalImage.dims();
    af::array resizedImage;
    resizedImage = af::resize(originalImage, output_size.first, output_size.second, AF_INTERP_BILINEAR);
    const af::dim4 resizedDims = resizedImage.dims();


    af::array boxes = in[3];
    af::array targetSize = in[1];
    if (!boxes.isempty()) {
      const float ratioWidth = float(resizedDims[0]) / float(originalDims[0]);
      const float ratioHeight = float(resizedDims[1]) / float(originalDims[1]);

      const std::vector<float> resizeVector = { ratioWidth, ratioHeight, ratioWidth, ratioHeight };
      af::array resizedArray = af::array(4, resizeVector.data());
      boxes = af::batchFunc(boxes, resizedArray, af::operator*);
    }

    std::vector<af::array> result =  { resizedImage, in[1], in[2], boxes, in[4] };
    return result;

  };
  return resizeCoco;
}


TransformAllFunction compose(std::vector<TransformAllFunction> fns) {
  return [fns](const std::vector<af::array>& in) {
    std::vector<af::array> out = in; 
    for(auto fn: fns) {
      out = fn(out);
    }
    return out;
  };
}

CocoDataset::CocoDataset(
    const std::string& list_file,
    std::vector<ImageTransform>& transformfns,
    int world_rank,
    int world_size,
    int batch_size,
    int num_threads,
    int prefetch_size,
    bool val
  ) {
  // Images
  const std::vector<std::string> filepaths = parseImageFilepaths(list_file);
  assert(filepaths.size() > 0);
  auto images = cocoDataLoader(filepaths);

  // Labels
  const std::vector<BBoxVector> bboxes = parseBoundingBoxes(list_file);
  auto bboxLabels = bboxLoader(bboxes);
  auto classLabels = classLoader(bboxes);
  auto labels =  merge({bboxLabels, classLabels});

  auto merged = merge({images, labels});

  std::shared_ptr<Dataset> transformed;

  transformed = merged;

  int maxSize = 1333;
  if (val) {
    transformed = std::make_shared<TransformAllDataset>(
         transformed, randomResize({800}, maxSize));
   } else {

     std::vector<int> scales = {480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800};
     TransformAllFunction trainTransform = randomSelect(
         {  
         randomResize(scales, maxSize),
         compose({
              randomResize({400, 500, 600}, -1),
              randomSizeCrop(384, 600),
              randomResize(scales, 1333)
          })
         }
      );

      transformed = std::make_shared<TransformAllDataset>(
           transformed, randomResize(scales, 1333));
   }

  transformed = std::make_shared<TransformAllDataset>(
      transformed, Normalize);

  transformed = std::make_shared<TransformDataset>(
      transformed, transformfns);

  auto next = transformed;
  if (!val) {
    shuffled_ = std::make_shared<ShuffleDataset>(next);
    next = shuffled_;
  }
  //auto next = transformed;
  //
  auto permfn = [world_size, world_rank](int64_t idx) {
    return (idx * world_size) + world_rank;
  };
  auto sampled = std::make_shared<ResampleDataset>(
    next, permfn, next->size() / world_size);

  auto prefetch = std::make_shared<PrefetchDataset>(sampled, num_threads, prefetch_size);
  //auto prefetch = sampled;
  batched_ = std::make_shared<BatchTransformDataset<CocoData>>(
      prefetch, batch_size, BatchDatasetPolicy::SKIP_LAST, cocoBatchFunc);

}

void CocoDataset::resample() {
  if(shuffled_) {
    shuffled_->resample();
  }
}


std::shared_ptr<Dataset> CocoDataset::getImages(
    const std::string list_file,
    std::vector<ImageTransform>& transformfns) {
  const std::vector<std::string> filepaths = parseImageFilepaths(list_file);
  auto images = cocoDataLoader(filepaths);
  return transform(images, transformfns);
}

std::shared_ptr<Dataset> CocoDataset::getLabels(std::string list_file) {
    const std::vector<BBoxVector> bboxes = parseBoundingBoxes(list_file);
    auto bboxLabels = bboxLoader(bboxes);
    auto classLabels = classLoader(bboxes);
    return merge({bboxLabels, classLabels});
}



int64_t CocoDataset::size() const {
  return batched_->size();
}

CocoData CocoDataset::get(const uint64_t idx) {
  return batched_->get(idx);
}

} // namespace objdet
} // namespace app
} // namespace fl
