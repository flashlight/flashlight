/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <algorithm>
#include <numeric>

namespace fl {

/** An implementation of edit distance meter, which measures the edit distance
 * between targets and predictions made by the model.
 * Example usage:
 *
 * \code
 * EditDistanceMeter meter();
 * for (auto& sample : data) {
 *   auto prediction = model(sample.input);
 *   meter.add(sample.target, prediction);
 * }
 * double letterErrorRate = meter.value()[0];
 * \endcode
 */
class EditDistanceMeter {
 public:
  /** A structure storing number of different type of errors when computing edit
   * distance. */
  struct ErrorState {
    int64_t ndel; //!< Number of deletion error
    int64_t nins; //!< Number of insertion error
    int64_t nsub; //!< Number of substitution error
    ErrorState() : ndel(0), nins(0), nsub(0) {}

    /** Sums up all the errors. */
    int64_t sum() const {
      return ndel + nins + nsub;
    }
  };

  /** Constructor of `EditDistanceMeter`. An instance will maintain five
   * counters initialized to 0:
   * - `n`: total target lengths
   * - `ndel`: total deletion error
   * - `nins`: total insertion error
   * - `nsub`: total substitution error
   */
  EditDistanceMeter();

  /** Computes edit distance between two arrayfire arrays `output` and `target`
   * and updates the counters.
   */
  void add(const af::array& output, const af::array& target);

  /** Updates all the counters with inputs sharing the same meaning. */
  void add(
      const int64_t n,
      const int64_t ndel,
      const int64_t nins,
      const int64_t nsub);

  /** Updates all the counters with an `ErrorState`. */
  void add(const ErrorState& es, const int64_t n) {
    add(n, es.ndel, es.nins, es.nsub);
  }

  /** Returns a vector of five values:
   * - `error rate`: \f$ \frac{(ndel + nins + nsub)}{n} \times 100.0 \f$
   * - `total length`: \f$ n \f$
   * - `deletion rate`: \f$ \frac{ndel}{n} \times 100.0\f$
   * - `insertion rate`: \f$ \frac{nins}{n} \times 100.0 \f$
   * - `substitution rate`: \f$ \frac{nsub}{n} \times 100.0 \f$
   */
  std::vector<double> value();

  /** Computes edit distance between two arrays `output` and `target`, with
   * length `olen` and `tlen` respectively, and updates the counters.
   */
  template <typename T, typename S>
  void
  add(const T& output, const S& target, const size_t olen, const size_t tlen) {
    auto err_state = levensteinDistance(output, target, olen, tlen);
    add(err_state, tlen);
  }

  /** Computes edit distance between two vectors `output` and `target`
   * and updates the counters.
   */
  template <typename T>
  void add(const std::vector<T>& output, const std::vector<T>& target) {
    add(output.data(), target.data(), output.size(), target.size());
  }

  /** Sets all the counters to 0. */
  void reset();

 private:
  int64_t n_;
  int64_t ndel_;
  int64_t nins_;
  int64_t nsub_;

  int64_t sumErr() {
    return ndel_ + nins_ + nsub_;
  }

  template <typename T>
  ErrorState levensteinDistance(
      const T& in1begin,
      const T& in2begin,
      size_t len1,
      size_t len2) const {
    std::vector<ErrorState> column(len1 + 1);
    for (int i = 0; i <= len1; ++i) {
      column[i].nins = i;
    }

    auto curin2 = in2begin;
    for (int x = 1; x <= len2; x++) {
      ErrorState lastdiagonal = column[0];
      column[0].ndel = x;
      auto curin1 = in1begin;
      for (int y = 1; y <= len1; y++) {
        auto olddiagonal = column[y];
        auto possibilities = {
            column[y].sum() + 1,
            column[y - 1].sum() + 1,
            lastdiagonal.sum() + ((*curin1 == *curin2) ? 0 : 1)};
        auto min_it =
            std::min_element(possibilities.begin(), possibilities.end());
        if (std::distance(possibilities.begin(), min_it) ==
            0) { // deletion error
          ++column[y].ndel;
        } else if (
            std::distance(possibilities.begin(), min_it) == 1) { // insertion
          // error
          column[y] = column[y - 1];
          ++column[y].nins;
        } else {
          column[y] = lastdiagonal;
          if (*curin1 != *curin2) { // substitution error
            ++column[y].nsub;
          }
        }

        lastdiagonal = olddiagonal;
        ++curin1;
      }
      ++curin2;
    }

    return column[len1];
  }
};
} // namespace fl
