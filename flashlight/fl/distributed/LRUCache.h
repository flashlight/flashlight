/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <list>
#include <unordered_map>

namespace fl {

namespace detail {

// The following section is taken from
// https://github.com/fairinternal/FAIR_rush/blob/master/cpid/distributed.h
template <typename K, typename V>
class LRUCache {
  // store keys of cache
  std::list<K> dq_;

  // store references of key in cache
  std::unordered_map<
      K,
      std::pair<typename std::list<K>::iterator, std::unique_ptr<V>>>
      map_;

  size_t csize_; // maximum capacity of cache

 public:
  explicit LRUCache(int n) : csize_(n) {}

  inline V* put(K k, std::unique_ptr<V>&& v) {
    if (map_.find(k) == map_.end()) {
      // Not in cache, cache size too big
      if (dq_.size() == csize_) {
        map_.erase(dq_.back());
        dq_.pop_back();
      }
    } else {
      dq_.erase(map_[k].first);
    }

    dq_.push_front(k);
    map_[k] = std::make_pair(dq_.begin(), std::move(v));
    return map_[k].second.get();
  }

  inline V* get(K const& k) {
    if (map_.find(k) == map_.end()) {
      return nullptr;
    } else {
      // Move list node to front
      auto& it = map_[k].first;
      dq_.splice(dq_.begin(), dq_, it);
      return map_[k].second.get();
    }
  }
};

inline void hashKeyHelper(std::stringstream&) {}

template <typename T, typename... Args>
inline void hashKeyHelper(std::stringstream& ss, const T& x, Args&&... params) {
  ss << " " << x;
  hashKeyHelper(ss, std::forward<Args>(params)...);
}

template <typename T, typename... Args>
inline std::string makeHashKey(T* ptr, Args&&... params) {
  std::stringstream ss;
  ss << typeid(T).name() << " " << reinterpret_cast<std::uintptr_t>(ptr);
  hashKeyHelper(ss, std::forward<Args>(params)...);
  return ss.str();
}

} // namespace detail

} // namespace fl
