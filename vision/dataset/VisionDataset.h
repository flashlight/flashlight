#pragma once 

#include "flashlight/dataset/Dataset.h"

namespace fl {

class VisionDataset : public Dataset {
  public:
    VisionDataset(std::vector<TransformFunction>& transformfns);

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

  static TransformFunction randomResizeCropTransform(
    const int resize,
    const float scaleLow,
    const float scaleHigh,
    const float ratioLow,
    const float ratioHigh);

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
  static TransformFunction centerCropTransform(const int size);

  /*
   * Flip an image horizontally with a probability @param p
   */
  static TransformFunction horizontalFlipTransform(const float p = 0.5);

  /*
   * Utility method for composing multiple transform functions
   */
  static TransformFunction compose(
      std::vector<TransformFunction>& transformfns);
  private:
      std::vector<TransformFunction>& transformfns_;
};

}
