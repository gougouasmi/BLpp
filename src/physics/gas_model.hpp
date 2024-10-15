#ifndef GAS_MODEL_H
#define GAS_MODEL_H

#include <cmath>

constexpr double GAM = 1.4;
constexpr double GAM1 = GAM - 1;
constexpr double GAM_GAM1 = GAM / GAM1;
constexpr double R_AIR = 296.92857142857144;
constexpr double R_AIR_INV = 1. / R_AIR;
constexpr double CP_AIR = R_AIR * GAM / GAM1;

#define AIR_CPG_RO(H, P) ((P)*GAM_GAM1 / (H))
#define AIR_CPG_DRO_DH(H, P) (-(P)*GAM_GAM1 / ((H) * (H)))

#define AIR_CPG_CP(H, P) (CP_AIR)

constexpr double SUTH_T0 = 273.;
constexpr double SUTH_T0_INV = 1. / SUTH_T0;

constexpr double SUTH_MU0 = 1.716e-5;
constexpr double SUTH_MU_S0 = 111.;
constexpr double SUTH_MU_POW_FACTOR_ = SUTH_MU0 * (SUTH_T0 + SUTH_MU_S0);

const double SUTH_MU_POW_FACTOR =
    SUTH_MU_POW_FACTOR_ * SUTH_T0_INV * sqrt(SUTH_T0_INV);

constexpr double SUTH_K0 = 0.0241;
constexpr double SUTH_K_S0 = 194;
constexpr double SUTH_K_POW_FACTOR_ = SUTH_K0 * (SUTH_T0 + SUTH_K_S0);

const double SUTH_K_POW_FACTOR =
    SUTH_K_POW_FACTOR_ * SUTH_T0_INV * sqrt(SUTH_T0_INV);

#define AIR_VISC(T) (SUTH_MU_POW_FACTOR * (T)*sqrt((T)) / ((T) + SUTH_MU_S0))
#define AIR_VISC_GRAD(T)                                                       \
  ((SUTH_MU_POW_FACTOR * sqrt((T)) / ((T) + SUTH_MU_S0)) *                     \
   (1.5 - (T) / ((T) + SUTH_MU_S0)))

#define AIR_COND(T) (SUTH_K_POW_FACTOR * (T)*sqrt((T)) / ((T) + SUTH_K_S0))
#define AIR_COND_GRAD(T)                                                       \
  ((SUTH_K_POW_FACTOR * sqrt((T)) / ((T) + SUTH_K_S0)) *                       \
   (1.5 - (T) / ((T) + SUTH_K_S0)))

#endif