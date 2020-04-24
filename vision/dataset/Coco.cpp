#include "vision/dataset/Coco.h"
#include "vision/dataset/Utils.h"

#include <arrayfire.h>

#include "iostream"

namespace {

using namespace fl::cv::dataset;

static const int kElementsPerBbox = 5;

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
      auto arr = af::array(num_bboxes, kElementsPerBbox, bbox.data());
      af_print(arr);
      return arr;
  });
}


}

namespace fl {
namespace cv {
namespace dataset {


std::shared_ptr<Dataset> coco(
    const std::string& list_file,
    std::vector<ImageTransform>& transformfns)  {

  const std::vector<std::string> filepaths = parseImageFilepaths(list_file);

  auto images = std::make_shared<FilepathLoader>(jpegLoader(filepaths));
  // TransformDataset will apply each transform in a vector to the respective 
  // af::array // Thus, we need to `compose` all of the transforms so are each
  // aplied
  std::vector<ImageTransform> transforms = { compose(transformfns) };
  auto transformed = std::make_shared<TransformDataset>(images, transforms);

  const std::vector<BBoxVector> bboxes = parseBoundingBoxes(list_file);
  auto labels = std::make_shared<BBoxLoader>(bboxLoader(bboxes));
  return std::make_shared<MergeDataset>(MergeDataset({transformed, labels}));
}

} // namespace dataset
} // namespace cv
} // namespace flashlight
