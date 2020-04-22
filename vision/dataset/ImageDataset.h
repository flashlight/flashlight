#pragma once

#include "vision/dataset/VisionDataset.h"

#include <unordered_map>

namespace fl {

/**
 * Creates a dataset from loadings JPEGs from files.
 * Example:
  \code{.cpp}
  // Make a dataset with 100 samples
  std::vector<std::string> fps = { "images/sample1.jpeg" };
  std::vector<uint64_t> labels = { 0 };
  std::vector<Dataset::TransformFunction> transforms = {
    ImageDataset::resizeTransform(224),
  };
  // Shuffle it
  ImageDataset imageds(fps, labels, transforms);
  std::cout << imageds.size() << "\n"; // 1;
  std::cout << image.get(0)[0].dims() << "\n"; // [224, 224, 3, 1];
  std::cout << image.get(0)[1].dims() << "\n"; // [1, 1, 1, 1];

 */
class ImageDataset : public VisionDataset {
 public:
  /**
   * Creates an `ImageDataset`.
   * @param[filepaths] filepaths to load images from
   * @param[labels] labels corresponding to the images in filepaths
   * @param[transformfns] Image transformations
   */
  ImageDataset(
      std::vector<std::string> filepaths,
      std::vector<TransformFunction>& transformfns);

  ImageDataset(std::string& dir);

  std::vector<af::array> get(const int64_t idx) const override;

  int64_t size() const override;


  static const uint64_t INPUT_IDX = 0;
  static const uint64_t TARGET_IDX = 1;

 private:
  std::shared_ptr<Dataset> ds_;
};

} // namespace fl
