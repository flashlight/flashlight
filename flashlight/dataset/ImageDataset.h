#pragma once

#include "flashlight/dataset/Dataset.h"

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
class ImageDataset : public Dataset {
 public:
  /**
   * Creates an `ImageDataset`.
   * @param[filepaths] filepaths to load images from
   * @param[labels] labels corresponding to the images in filepaths
   * @param[transformfns] Image transformations
   */
  ImageDataset(
      std::vector<std::string> filepaths,
      std::vector<uint64_t> labels,
      std::vector<TransformFunction>& transformfns);

  std::vector<af::array> get(const int64_t idx) const override;

  int64_t size() const override;

  static TransformFunction normalizeImage(
      const std::vector<float>& mean_,
      const std::vector<float>& std_);

  /*
   * Randomly resize the image between sizes
   * @param low
   * @param high
   * This transform helps to create scale invariance
   */
  static TransformFunction randomResizeTransform(const int low, const int high);

  /*
   * Randomly crop an image with target height of @param th and a target width of 
   * @params tw
   */
  static TransformFunction randomCropTransform(const int th, const int tw);

  /*
   * Resize the shortest edge of the image to size @param resize
   */
  static TransformFunction resizeTransform(const uint64_t resize);

  /*
   * Take a center crop of an image so its size is @param size
   */
  static TransformFunction centerCrop(const int size);

  /*
   * Flip an image horizontally with a probability @param p
   */
  static TransformFunction horizontalFlipTransform(const float p = 0.5);

  /*
   * Utility method for composing multiple transform functions
   */
  static TransformFunction compose(
      std::vector<TransformFunction>& transformfns);

  static const uint64_t INPUT_IDX = 0;
  static const uint64_t TARGET_IDX = 1;

 private:
  std::shared_ptr<Dataset> ds_;
};

} // namespace fl
