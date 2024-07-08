#ifndef GAS_MODEL_H
#define GAS_MODEL_H

#include <cmath>

const double GAM = 1.4;
const double GAM1 = GAM - 1;
const double R_AIR = 296.92857142857144;
const double CP_AIR = R_AIR * GAM / GAM1;

#define AIR_CPG_RO(H, P) ((P)*GAM / (GAM1 * (H)))
#define AIR_CPG_CP(H, P) (CP_AIR)

const double SUTH_MU0 = 1.716e-5;
const double SUTH_T0 = 273.;
const double SUTH_MU_S0 = 111.;

const double SUTH_K0 = 0.0241;
const double SUTH_K_S0 = 194;

#define AIR_VISC(T)                                                            \
  (SUTH_MU0 * pow((T) / SUTH_T0, 1.5) * (SUTH_T0 + SUTH_MU_S0) /               \
   ((T) + SUTH_MU_S0))
#define AIR_COND(T)                                                            \
  (SUTH_K0 * pow((T) / SUTH_T0, 1.5) * (SUTH_T0 + SUTH_K_S0) /                 \
   ((T) + SUTH_K_S0))

#endif