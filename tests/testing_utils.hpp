#ifndef TESTING_UTILS_HPP
#define TESTING_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "generic_vector.hpp"

bool inline isClose(double a, double b, double rel_tol = 1e-9) {
  return abs(a - b) <= rel_tol * std::fmax(std::fabs(a), std::fabs(b));
}

bool inline allClose(const std::vector<double> &vec1,
                     const std::vector<double> &vec2, int size,
                     double rel_tol = 1e-9) {
  assert(vec1.size() >= size);
  assert(vec2.size() >= size);

  double error = 0., vec1_norm = 0., vec2_norm = 0.;
  for (int id = 0; id < size; id++) {
    error += pow(vec1[id] - vec2[id], 2.0);
    vec1_norm += pow(vec1[id], 2.0);
    vec2_norm += pow(vec2[id], 2.0);
  }

  vec1_norm = sqrt(vec1_norm);
  vec2_norm = sqrt(vec2_norm);
  error = sqrt(error);

  return error <= rel_tol * std::fmax(vec1_norm, vec2_norm);
}

void inline fillWithRandomData(std::vector<double> &data, int size) {
  double denom = 1. / static_cast<double>(RAND_MAX);
  for (int i = 0; i < size; ++i) {
    int random_val = rand();
    data[i] = static_cast<double>(random_val) * denom;
  }
}

void inline print_matrix_column_major(std::vector<double> &matrix_data,
                                      int xdim, int ydim) {
  int offset = 0;
  std::cout << "[\n";
  for (int col_id = 0; col_id < ydim; col_id++) {
    std::cout << " [";
    for (int row_id = 0; row_id < xdim; row_id++) {
      std::cout << matrix_data[offset + row_id] << ", ";
    }
    std::cout << "],\n";
    offset += xdim;
  }
  std::cout << "].\n";
}

void inline print_matrix_row_major(std::vector<double> &matrix_data, int xdim,
                                   int ydim) {
  int offset = 0;
  std::cout << "[\n";
  for (int row_id = 0; row_id < xdim; row_id++) {
    std::cout << " [";
    for (int col_id = 0; col_id < ydim; col_id++) {
      std::cout << matrix_data[offset + col_id] << ", ";
    }
    std::cout << "],\n";
    offset += ydim;
  }
  std::cout << "].\n";
}

void inline print_vec(std::vector<double> &vec, int size) {
  std::cout << "[";
  for (const double &val : vec) {
    std::cout << val << ", ";
  }
  std::cout << "].\n";
}

namespace Generic {

template <typename T, std::size_t ctime_x1dim = 0, std::size_t ctime_x2dim = 0>
bool allClose(const Generic::Vector<T, ctime_x1dim> &arr1,
              const Generic::Vector<T, ctime_x2dim> &arr2, T rel_tol = 1e-9) {
  const size_t xdim = arr1.size();
  assert(xdim == arr2.size());

  T error = 0., arr1_norm = 0., arr2_norm = 0.;
  for (int id = 0; id < xdim; id++) {
    error += pow(arr1[id] - arr2[id], 2.0);
    arr1_norm += pow(arr1[id], 2.0);
    arr2_norm += pow(arr2[id], 2.0);
  }

  arr1_norm = sqrt(arr1_norm);
  arr2_norm = sqrt(arr2_norm);
  error = sqrt(error);

  T max_error = std::fmax(arr1_norm, arr2_norm);

  bool close_enough = error <= rel_tol * max_error;

  return close_enough;
}

template <typename T, std::size_t ctime_xdim = 0>
void inline fillWithRandomData(Generic::Vector<T, ctime_xdim> &data, int size) {
  assert(size <= data.size());
  T denom = 1. / static_cast<T>(RAND_MAX);
  for (int i = 0; i < size; ++i) {
    int random_val = rand();
    data[i] = static_cast<T>(random_val) * denom;
  }
}

template <typename T, std::size_t ctime_xdim = 0>
void inline print_vec(Generic::Vector<T, ctime_xdim> &vec, int xdim) {
  std::cout << "[";
  for (const T &val : vec) {
    std::cout << val << ", ";
  }
  std::cout << "].\n";
}

} // namespace Generic

#endif