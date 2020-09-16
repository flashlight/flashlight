#pragma once

namespace fl {

class CocoDataset : public Dataset {
  public:

  CocoDataset(
      std::string lst, 
      std::vector<TransformFunction>& transformfns)
  
  std::vector<af::array> get(const int64_t idx) const override;

  int64_t size() const override;
}

}
