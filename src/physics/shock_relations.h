#ifndef SHOCK_RELATIONS_H
#define SHOCK_RELATIONS_H

#include <cmath>

typedef struct ShockRatios {
  double pressure;
  double velocity;
  double density;
  ShockRatios(double p_ratio, double u_ratio, double ro_ratio)
      : pressure(p_ratio), velocity(u_ratio), density(ro_ratio){};
} ShockRatios;

ShockRatios cpg_shock_ratios(double mach, double gamma = 1.4,
                             double beta = 0.5 * M_PI) {

  double gamp1 = gamma + 1;
  double gamm1 = gamma - 1;

  double mach2 = mach * mach;
  double sin_beta = sin(beta);
  double sin2_beta = sin_beta * sin_beta;
  double cos_beta = cos(beta);

  double p_ratio = 1. + 2.0 * gamma / gamp1 * (mach2 * sin2_beta - 1.0);

  double ro_ratio =
      (gamp1 * mach2 * sin2_beta) / (gamm1 * mach2 * sin2_beta + 2.0);

  double u2_V1 = 1.0 - 2.0 * (mach2 * sin2_beta - 1.0) / ((gamp1)*mach2);
  double v2_V1 = 2.0 * (mach2 * sin2_beta - 1.0) * cos(beta) / (gamp1 * mach2);

  double u_ratio = sqrt(u2_V1 * u2_V1 + v2_V1 * v2_V1);

  return ShockRatios(p_ratio, u_ratio, ro_ratio);
}

#endif