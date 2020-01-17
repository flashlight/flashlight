#pragma once

#include "flashlight/dataset/Dataset.h"

namespace fl {

/**
 * Dataset created by loading jpegs from a directory.
 * The jpegs must be placed in subdirectories representing their class in a
 * similar fashion to imagenet.
 *
 * For example
 * train/
 * >> n01440764/
 * >>>> n01440764_10026.JPEG
 * >>>> n01440764_10027.JPEG
 * val/
 * >> n01440764
 * >>>> ILSVRC2012_val_00000293.JPEG
 * >>>> ILSVRC2012_val_00002138.JPEG
 * ...
 * labels.txt
 * ....
 * n01440764,tench
 * n01443537,goldfish
 * n01484850,great white shark
 * n01491361,tiger shark
 * .....
 *
 */
class ImageDataset : public fl::Dataset {
 public:
  /*
   * Creates an `ImageDataset` by loading images in @param[sample_dir] and
   * determines their labels using @params[labels]. We resize the images using
   * @param[resize] \code{.cpp} std::string imagenet_base = "/data/imagenet/";
   * auto labels = ImageDataset::parseLabels(imagenet_base + "labels.txt");
   * ds = ImageDataset(imagenet_base + "train", labels);
   * auto sample = ds.get(0)
   * std::cout << sample[0].dims() << std::endl; // {244, 244, 3, 1}
   * std::cout << sample[1].dims() << std::endl; // {1, 1, 1, 1}
   *
   */
  ImageDataset(
      const std::string& sample_dir,
      const std::unordered_map<std::string, uint32_t>& labels,
      uint64_t resize = 224);

  std::vector<af::array> get(const int64_t idx) const override;

  int64_t size() const override;

  static std::unordered_map<std::string, uint32_t> parseLabels(
      const std::string& label_file);

  static void normalizeImage(af::array& in);

  static const uint64_t INPUT_IDX = 0;

  static const uint64_t TARGET_IDX = 1;

 private:
  std::vector<std::string> filepaths_;
  std::vector<uint64_t> labels_;
  const uint64_t resize_;
};

} // namespace fl
