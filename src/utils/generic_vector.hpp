#ifndef GENERIC_VECTOR_HPP
#define GENERIC_VECTOR_HPP

#include <array>
#include <type_traits>
#include <vector>

template <typename T, std::size_t ctime_dim = 0>
using GenericVector = std::conditional_t<ctime_dim == 0, std::vector<T>,
                                         std::array<T, ctime_dim>>;

#endif