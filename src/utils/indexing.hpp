#ifndef INDEXING_H
#define INDEXING_H

#include <array>

template <size_t N>
constexpr bool complete_indexing(const std::array<int, N> &indices) {
  bool index_coverage[N];

  for (auto &val : index_coverage) {
    val = false;
  }

  for (const int &index : indices) {
    if (index >= 0 && index < N) {
      index_coverage[index] = true;
    }
  }

  for (const bool &index_found : index_coverage) {
    if (!index_found)
      return false;
  }

  return true;
}

#endif