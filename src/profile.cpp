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

double compute_rhs_default(std::vector<double> &state, std::vector<double> &rhs,
                           int offset, ProfileParams &params) {
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

double compute_rhs_cpg(std::vector<double> &state, std::vector<double> &rhs,
                       int offset, ProfileParams &params) {
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