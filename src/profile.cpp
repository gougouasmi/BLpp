#include "profile.h"
#include "gas_model.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

/*
 * Evolve the BL equations from an initial guess until it
 * converges and return the resulting profile.
 *
 */

void initialize_default(ProfileParams &profile_params,
                        std::vector<double> &state) {
  double fpp0 = profile_params.fpp0;
  double gp0 = profile_params.gp0;
  double g0 = profile_params.g0;

  double romu0 = 1.;
  double prandtl0 = 1.;

  switch (profile_params.wall_type) {
  case WallType::Wall:
    state[FPP_ID] = romu0 * fpp0;
    state[FP_ID] = 0.0;
    state[F_ID] = 0.0;
    state[GP_ID] = (romu0 / prandtl0) * gp0;
    state[G_ID] = profile_params.g0;
    break;
  case WallType::Adiabatic:
    state[FPP_ID] = romu0 * fpp0;
    state[FP_ID] = 0.0;
    state[F_ID] = 0.0;
    state[GP_ID] = 0.0;
    state[G_ID] = g0;
    break;
  }
}

void initialize_cpg(ProfileParams &profile_params, std::vector<double> &state) {
  double fpp0 = profile_params.fpp0;
  double gp0 = profile_params.gp0;
  double g0 = profile_params.g0;

  double edge_pressure = profile_params.pe;
  double edge_enthalpy = profile_params.he;

  double edge_density = AIR_CPG_RO(edge_enthalpy, edge_pressure);
  double edge_temperature = edge_pressure / (edge_density * R_AIR);

  double edge_visc = AIR_VISC(edge_temperature);

  profile_params.roe = edge_density;
  profile_params.mue = edge_visc;
  profile_params.eckert = pow(profile_params.ue, 2) / profile_params.he;

  //
  double density = AIR_CPG_RO(g0 * edge_enthalpy, edge_pressure);
  double cp = AIR_CPG_CP(g0 * edge_enthalpy, edge_pressure);
  double temperature = edge_pressure / (density * R_AIR);

  double visc = AIR_VISC(temperature);
  double cond = AIR_COND(temperature);

  double romu0 = (density * visc) / (edge_density * edge_visc);
  double prandtl0 = visc * cp / cond;

  switch (profile_params.wall_type) {
  case WallType::Wall:
    state[FPP_ID] = romu0 * fpp0;
    state[FP_ID] = 0.0;
    state[F_ID] = 0.0;
    state[GP_ID] = (romu0 / prandtl0) * gp0;
    state[G_ID] = g0;
    break;
  case WallType::Adiabatic:
    state[FPP_ID] = romu0 * fpp0;
    state[FP_ID] = 0.0;
    state[F_ID] = 0.0;
    state[GP_ID] = 0.0;
    state[G_ID] = g0;
    break;
  }
}

double compute_rhs_default(const std::vector<double> &state,
                           std::vector<double> &rhs, int offset,
                           ProfileParams &params) {
  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[offset + FP_ID];
  double g = state[offset + G_ID];

  double fpp = state[offset + FPP_ID] / romu;
  double f = state[offset + F_ID];
  double gp = state[offset + GP_ID] / romu * prandtl;

  rhs[FPP_ID] = -f * fpp;
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

double compute_rhs_cpg(const std::vector<double> &state,
                       std::vector<double> &rhs, int offset,
                       ProfileParams &params) {
  //
  double fp = state[offset + FP_ID];
  double g = state[offset + G_ID];

  double pe = params.pe;
  double he = params.he;

  // ro, temperature, cp = thermo_fun(pe, he * g)
  // mu, k = transport_fun(temperature)
  double ro = AIR_CPG_RO(g * he, pe);
  double cp = AIR_CPG_CP(g * he, pe);
  double temperature = pe / (ro * R_AIR);

  double mu = AIR_VISC(temperature);
  double k = AIR_COND(temperature);

  double roe = params.roe;
  double mue = params.mue;

  double romu = (ro * mu) / (roe * mue);
  double prandtl = mu * cp / k;
  double eckert = params.eckert;

  double fpp = state[offset + FPP_ID] / romu;
  double f = state[offset + F_ID];
  double gp = state[offset + GP_ID] / romu * prandtl;

  rhs[FPP_ID] = -f * fpp;
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

void compute_rhs_jacobian_default(const std::vector<double> &state,
                                  std::vector<double> &matrix_data,
                                  ProfileParams &params) {
  assert(matrix_data.size() == FLAT_PLATE_RANK * FLAT_PLATE_RANK);

  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[FP_ID];
  double g = state[G_ID];

  double fpp = state[FPP_ID] / romu;
  double f = state[F_ID];
  double gp = state[GP_ID] / romu * prandtl;

  int offset;

  // rhs[FPP_ID] = - f * fpp
  offset = FPP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -f / romu;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[FP_ID] = fpp
  offset = FP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 1. / romu;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[F_ID] = fp
  offset = F_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp)
  offset = GP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * prandtl / romu;
  matrix_data[offset + G_ID] = 0.;

  // rhs[G_ID] = gp;
  offset = G_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = prandtl / romu;
  matrix_data[offset + G_ID] = 0.;
}

void compute_rhs_jacobian_cpg(const std::vector<double> &state,
                              std::vector<double> &matrix_data,
                              ProfileParams &params) {
  assert(matrix_data.size() == FLAT_PLATE_RANK * FLAT_PLATE_RANK);

  double fp = state[FP_ID];
  double g = state[G_ID];

  double pe = params.pe;
  double he = params.he;

  // ro, temperature, cp = thermo_fun(pe, he * g)
  // mu, k = transport_fun(temperature)
  double ro = AIR_CPG_RO(g * he, pe);
  double dro_dg = AIR_CPG_DRO_DH(g * he, pe) * he;

  double cp = AIR_CPG_CP(g * he, pe);

  double temperature = pe / (ro * R_AIR);
  double dtemp_dg = -pe * dro_dg / (ro * ro * R_AIR);

  double mu = AIR_VISC(temperature);
  double k = AIR_COND(temperature);

  double dmu_dg = AIR_VISC_GRAD(temperature) * dtemp_dg;
  double dk_dg = AIR_COND_GRAD(temperature) * dtemp_dg;

  double roe = params.roe;
  double mue = params.mue;

  double romu = (ro * mu) / (roe * mue);
  double dromu_dg = (dro_dg * mu + ro * dmu_dg) / (roe * mue);

  double prandtl = mu * cp / k;
  double dprandtl_dg = (dmu_dg / k - mu * dk_dg / (k * k)) * cp;

  double eckert = params.eckert;

  double fpp = state[FPP_ID] / romu;
  double dfpp_dfpp = 1. / romu;
  double dfpp_dg = -dromu_dg * fpp / romu;

  double f = state[F_ID];

  double gp = state[GP_ID] / romu * prandtl;
  double dgp_dgp = prandtl / romu;
  double dgp_dg =
      state[GP_ID] * (dprandtl_dg / romu - dromu_dg * prandtl / (romu * romu));

  int offset;

  // rhs[FPP_ID] = -f * fpp;
  offset = FPP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -f * dfpp_dfpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = -f * dfpp_dg;

  // rhs[FP_ID] = fpp;
  offset = FP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = dfpp_dfpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  offset = F_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  offset = GP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * dgp_dgp;
  matrix_data[offset + G_ID] = -(
      f * dgp_dg + eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp));

  // rhs[G_ID] = gp;
  offset = G_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = dgp_dgp;
  matrix_data[offset + G_ID] = dgp_dg;
}