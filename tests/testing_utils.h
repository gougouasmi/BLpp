#ifndef TESTING_UTILS_H
#define TESTING_UTILS_H

#include <algorithm>
#include <cmath>

bool isClose(double a, double b, double rel_tol = 1e-9) {
  return abs(a - b) <= rel_tol * std::fmax(std::fabs(a), std::fabs(b));
}

#endif