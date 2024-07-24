#ifndef TESTING_UTILS_H
#define TESTING_UTILS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

void fillWithRandomData(std::vector<double> &data, int size) {
  double denom = 1. / static_cast<double>(RAND_MAX);
  for (int i = 0; i < size; ++i) {
    int random_val = rand();
    data[i] = static_cast<double>(random_val) * denom;
  }
}

void print_matrix(std::vector<double> &matrix_data, int xdim) {
  int offset = 0;
  std::cout << "[\n";
  for (int row_id = 0; row_id < xdim; row_id++) {
    std::cout << " [";
    for (int col_id = 0; col_id < xdim; col_id++) {
      std::cout << matrix_data[offset + col_id] << ", ";
    }
    std::cout << "],\n";
    offset += xdim;
  }
  std::cout << "].\n";
}

void print_vec(std::vector<double> &vec, int size) {
  std::cout << "[";
  for (const double &val : vec) {
    std::cout << val << ", ";
  }
  std::cout << "].\n";
}

#endif