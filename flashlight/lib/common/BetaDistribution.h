
/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright (c) 2013 Joseph Mansfield

#include <iostream>
#include <random>
#include <sstream>
#include <string>

namespace fl {
namespace lib {

/**
 * A simple C++11 Beta Distribution implementation.
 * Source - https://gist.github.com/sftrabbit/5068941
 *
 * It conforms to the requirements for a random number distribution at
 * [here](https://en.cppreference.com/w/cpp/named_req/RandomNumberDistribution)
 * and is therefore compatible with the <random> C++ library header. Naming
 * format also follows the convention of <random> C++ library.
 *
 * This implementation utilized `std::gamma_distribution`, given that
 * X / (X + Y) ~ Beta(a, b), when X ~ Gamma(a, c), Y ~ Gamma(b, c).
 */
template <typename RealType = double>
class beta_distribution {
 public:
  using result_type = RealType;

  class param_type {
   public:
    using distribution_type = beta_distribution;

    explicit param_type(RealType a = 1.0, RealType b = 1.0)
        : a_param(a), b_param(b) {}

    RealType a() const {
      return a_param;
    }
    RealType b() const {
      return b_param;
    }

    bool operator==(const param_type& other) const {
      return (a_param == other.a_param && b_param == other.b_param);
    }

    bool operator!=(const param_type& other) const {
      return !(*this == other);
    }

   private:
    RealType a_param, b_param;
  };

  explicit beta_distribution(RealType a = 1.0, RealType b = 1.0)
      : a_gamma(a), b_gamma(b) {}
  explicit beta_distribution(const param_type& param)
      : a_gamma(param.a()), b_gamma(param.b()) {}

  void reset() {}

  param_type param() const {
    return param_type(a(), b());
  }

  void param(const param_type& param) {
    a_gamma = gamma_dist_type(param.a());
    b_gamma = gamma_dist_type(param.b());
  }

  template <typename URNG>
  result_type operator()(URNG& engine) {
    return generate(engine, a_gamma, b_gamma);
  }

  template <typename URNG>
  result_type operator()(URNG& engine, const param_type& param) {
    gamma_dist_type a_param_gamma(param.a()), b_param_gamma(param.b());
    return generate(engine, a_param_gamma, b_param_gamma);
  }

  result_type min() const {
    return 0.0;
  }
  result_type max() const {
    return 1.0;
  }

  RealType a() const {
    return a_gamma.alpha();
  }
  RealType b() const {
    return b_gamma.alpha();
  }

  bool operator==(const beta_distribution<result_type>& other) const {
    return (
        param() == other.param() && a_gamma == other.a_gamma &&
        b_gamma == other.b_gamma);
  }

  bool operator!=(const beta_distribution<result_type>& other) const {
    return !(*this == other);
  }

 private:
  using gamma_dist_type = std::gamma_distribution<result_type>;

  gamma_dist_type a_gamma, b_gamma;

  template <typename URNG>
  result_type
  generate(URNG& engine, gamma_dist_type& x_gamma, gamma_dist_type& y_gamma) {
    result_type x = x_gamma(engine);
    return x / (x + y_gamma(engine));
  }
};

template <typename CharT, typename RealType>
std::basic_ostream<CharT>& operator<<(
    std::basic_ostream<CharT>& os,
    const beta_distribution<RealType>& beta) {
  os << "~Beta(" << beta.a() << "," << beta.b() << ")";
  return os;
}

template <typename CharT, typename RealType>
std::basic_istream<CharT>& operator>>(
    std::basic_istream<CharT>& is,
    beta_distribution<RealType>& beta) {
  std::string str;
  RealType a, b;
  if (std::getline(is, str, '(') && str == "~Beta" && is >> a &&
      is.get() == ',' && is >> b && is.get() == ')') {
    beta = beta_distribution<RealType>(a, b);
  } else {
    is.setstate(std::ios::failbit);
  }
  return is;
}

} // namespace lib
} // namespace fl
