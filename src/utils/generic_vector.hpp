#ifndef GENERIC_VECTOR_HPP
#define GENERIC_VECTOR_HPP

#include <array>
#include <type_traits>
#include <vector>

namespace Generic {

template <typename T, std::size_t ctime_dim = 0>
using Vector = std::conditional_t<ctime_dim == 0, std::vector<T>,
                                  std::array<T, ctime_dim>>;

template <std::size_t ctime_dim = 0>
double inline VectorNorm(const Generic::Vector<double, ctime_dim> &x) {
  double out = 0.;
  for (int idx = 0; idx < x.size(); idx++) {
    out += x[idx] * x[idx];
  }
  return sqrt(out);
}

} // namespace Generic

#endif