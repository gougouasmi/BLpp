#ifndef STAGNATION_HPP
#define STAGNATION_HPP

#include <cmath>

void inline ComputeStagnationRatios(const double mach, const double gamma,
                                    double &temp_ratio, double &density_ratio,
                                    double &pressure_ratio) {
  temp_ratio = (1. + 0.5 * (gamma - 1) * mach * mach);
  density_ratio = pow(temp_ratio, 1. / (gamma - 1.));
  pressure_ratio = pow(temp_ratio, gamma / (gamma - 1.));
}

#endif