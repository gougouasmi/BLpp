#ifndef TESTING_UTILS_H
#define TESTING_UTILS_H

#include <algorithm>
#include <cassert>
#include <cmath>
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

#endif