#ifndef PROFILE_FUNCTIONS_CPG_HPP
#define PROFILE_FUNCTIONS_CPG_HPP

#include "bl_model_struct.hpp"
#include "gas_model.hpp"
#include "profile_struct.hpp"

using std::vector;

inline void initialize_cpg(ProfileParams &profile_params,
                           vector<double> &state) {
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

template <std::size_t CTIME_RANK, std::size_t TARGET_RANK>
void initialize_sensitivity_cpg(
    ProfileParams &profile_params,
    Generic::Vector<double, CTIME_RANK * TARGET_RANK> &state_sensitivity_cm) {
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

  // Gradient with respect to g0
  double density_grad =
      AIR_CPG_DRO_DH(g0 * edge_enthalpy, edge_pressure) * edge_enthalpy;

  double temperature_grad =
      -edge_pressure * density_grad / (density * density * R_AIR);

  double visc_grad = AIR_VISC_GRAD(temperature) * temperature_grad;

  double romu_gradient =
      (density_grad * visc + density * visc_grad) / (edge_density * edge_visc);

  //

  std::fill(state_sensitivity_cm.begin(), state_sensitivity_cm.end(), 0.);

  switch (profile_params.wall_type) {
  case WallType::Wall:
    // state[FPP_ID] = romu0 * fpp0;
    // state[FP_ID] = 0.0;
    // state[F_ID] = 0.0;
    // state[GP_ID] = (romu0 / prandtl0) * gp0;
    // state[G_ID] = g0;

    // Sensitivities wrt f''(0)
    state_sensitivity_cm[FPP_ID] = romu0;

    // Sensitivities wrt g'(0)
    state_sensitivity_cm[BL_RANK + GP_ID] = romu0 / prandtl0;

    break;
  case WallType::Adiabatic:
    // state[FPP_ID] = romu0 * fpp0;
    // state[FP_ID] = 0.0;
    // state[F_ID] = 0.0;
    // state[GP_ID] = 0.0;
    // state[G_ID] = g0;

    // Sensitivities wrt f''(0)
    state_sensitivity_cm[FPP_ID] = romu0;

    // Sensitivities wrt g(0)
    state_sensitivity_cm[BL_RANK + FPP_ID] = romu_gradient * fpp0;
    state_sensitivity_cm[BL_RANK + G_ID] = 1.;
  }
}

template <std::size_t CTIME_RANK>
double limit_update_cpg(const Generic::Vector<double, CTIME_RANK> &state,
                        const Generic::Vector<double, CTIME_RANK> &state_varn,
                        const ProfileParams &params) {
  double alpha = 1.;

  // Do not let u become negative
  if (state_varn[FP_ID] < 0) {
    alpha = std::min(alpha, 0.2 * state[FP_ID] / (-state_varn[FP_ID]));
  } else {
    alpha = std::min(alpha,
                     0.2 * (1.2 - state[FP_ID]) / (state_varn[FP_ID] + 1e-30));
  }

  if (state_varn[FPP_ID] < 0) {
    alpha =
        std::min(alpha, 0.2 * (state[FPP_ID]) / (-state_varn[FPP_ID] + 1e-30));
  }
  alpha = std::min(alpha, 0.2 * state[G_ID] / fabs(state_varn[G_ID] + 1e-30));

  return alpha;
}

template <std::size_t CTIME_RANK>
double compute_rhs_cpg(const Generic::Vector<double, CTIME_RANK> &state,
                       int state_offset, const vector<double> &field,
                       int field_offset,
                       Generic::Vector<double, CTIME_RANK> &rhs,
                       const ProfileParams &params) {
  //
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

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

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  rhs[FPP_ID] = -f * fpp;
  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

template <std::size_t CTIME_RANK>
double compute_lsim_rhs_cpg(const Generic::Vector<double, CTIME_RANK> &state,
                            int state_offset, const vector<double> &field,
                            int field_offset,
                            Generic::Vector<double, CTIME_RANK> &rhs,
                            const ProfileParams &params) {
  //
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

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

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  double c1 = params.c1; // 2. * (xi / ue) * due_dxi;
  double c2 = params.c2; // 2. * xi * dhe_dxi / he;
  double c3 = params.c3; // 2. * xi * ue * due_dxi / he;

  // printf("c1 = %.2e, c2 = %.2e, c3 = %.2e.\n", c1, c2, c3);

  rhs[FPP_ID] = -f * fpp + c1 * (fp * fp - roe / ro);
  rhs[GP_ID] =
      -(f * gp + romu * eckert * fpp * fpp) + fp * (c2 * g + c3 * (roe / ro));
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

template <std::size_t CTIME_RANK>
double compute_full_rhs_cpg(const Generic::Vector<double, CTIME_RANK> &state,
                            int state_offset, const vector<double> &field,
                            int field_offset,
                            Generic::Vector<double, CTIME_RANK> &rhs,
                            const ProfileParams &params) {
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

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

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  double c1 = params.c1; // 2. * (xi / ue) * due_dxi;
  double c2 = params.c2; // 2. * xi * dhe_dxi / he;
  double c3 = params.c3; // 2. * xi * ue * due_dxi / he;

  // momemtum equation coefficients
  double m0 = field[field_offset + FIELD_M0_ID];
  double m1 = field[field_offset + FIELD_M1_ID];
  double s0 = field[field_offset + FIELD_S0_ID];
  double s1 = field[field_offset + FIELD_S1_ID];

  // energy equation coefficients
  double e0 = field[field_offset + FIELD_E0_ID];
  double e1 = field[field_offset + FIELD_E1_ID];

  rhs[FPP_ID] = -f * fpp + c1 * (fp * fp - roe / ro) + fp * (m0 * fp + m1) -
                fpp * (s0 * f + s1);
  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp) +
               fp * (c2 * g + c3 * (roe / ro)) + fp * (e0 * g + e1) -
               gp * (s0 * f + s1);
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

template <std::size_t CTIME_RANK>
void compute_rhs_jacobian_cpg(const Generic::Vector<double, CTIME_RANK> &state,
                              int state_offset, const vector<double> &field,
                              int field_offset,
                              Generic::Vector<double, CTIME_RANK> &matrix_data,
                              const ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

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

  double fpp = state[state_offset + FPP_ID] / romu;
  double dfpp_dfpp = 1. / romu;
  double dfpp_dg = -dromu_dg * fpp / romu;

  double f = state[state_offset + F_ID];

  double gp = state[state_offset + GP_ID] / romu * prandtl;
  double dgp_dgp = prandtl / romu;
  double dgp_dg = state[state_offset + GP_ID] *
                  (dprandtl_dg / romu - dromu_dg * prandtl / (romu * romu));

  int offset;

  // rhs[FPP_ID] = -f * fpp;
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f * dfpp_dfpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + G_ID] = -f * dfpp_dg;

  // rhs[FP_ID] = fpp;
  offset = FP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = dfpp_dfpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  offset = F_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[offset + GP_ID] = -f * dgp_dgp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + G_ID] = -(
      f * dgp_dg + eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp));

  // rhs[G_ID] = gp;
  offset = G_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0.;
  matrix_data[offset + GP_ID] = dgp_dgp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + G_ID] = dgp_dg;
}

template <std::size_t CTIME_RANK>
void compute_lsim_rhs_jacobian_cpg(
    const Generic::Vector<double, CTIME_RANK> &state, int state_offset,
    const vector<double> &field, int field_offset,
    Generic::Vector<double, CTIME_RANK> &matrix_data,
    const ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];
  double g1 = 1. / g;

  double pe = params.pe;
  double he = params.he;

  // ro, temperature, cp = thermo_fun(pe, he * g)
  // mu, k = transport_fun(temperature)
  double ro = (pe * GAM_GAM1 / he) * g1; // AIR_CPG_RO(g * he, pe);
  double ro1 = 1. / ro;
  double dro_dg = -ro * g1; // AIR_CPG_DRO_DH(g * he, pe) * he;

  double cp = CP_AIR;

  double temperature = pe * ro1 * R_AIR_INV;
  double dtemp_dg = -pe * dro_dg * (ro1 * ro1 * R_AIR_INV);

  double mu = AIR_VISC(temperature);
  double k = AIR_COND(temperature);

  double dmu_dg = AIR_VISC_GRAD(temperature) * dtemp_dg;
  double dk_dg = AIR_COND_GRAD(temperature) * dtemp_dg;

  double roe = params.roe;
  double mue = params.mue;

  double romu = (ro * mu) / (roe * mue);
  double romu1 = 1. / romu;
  double dromu_dg = (dro_dg * mu + ro * dmu_dg) / (roe * mue);

  double prandtl = mu * cp / k;
  double dprandtl_dg = (dmu_dg / k - mu * dk_dg / (k * k)) * cp;

  double eckert = params.eckert;

  double fpp = state[state_offset + FPP_ID] * romu1;
  double dfpp_dfpp = romu1;
  double dfpp_dg = -dromu_dg * fpp * romu1;

  double f = state[state_offset + F_ID];

  double gp = state[state_offset + GP_ID] * romu1 * prandtl;
  double dgp_dgp = prandtl * romu1;
  double dgp_dg = state[state_offset + GP_ID] * romu1 *
                  (dprandtl_dg - dromu_dg * prandtl * romu1);

  double c1 = params.c1; // 2. * (xi / ue) * due_dxi;
  double c2 = params.c2; // 2. * xi * dhe_dxi / he;
  double c3 = params.c3; // 2. * xi * ue * due_dxi / he;

  int offset;

  // rhs[FPP_ID] = -f * fpp + c1 * (fp * fp - roe / ro);
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f * dfpp_dfpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + FP_ID] = 2. * c1 * fp;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + G_ID] = -f * dfpp_dg + c1 * (roe * dro_dg * (ro1 * ro1));

  // rhs[GP_ID] =
  //   -(f * gp + eckert * romu * fpp * fpp) +
  //   fp * (c2 * g + c3 * (roe / ro));
  //
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[offset + GP_ID] = -f * dgp_dgp;
  matrix_data[offset + FP_ID] = (c2 * g + c3 * (roe * ro1));
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + G_ID] =
      -(f * dgp_dg +
        eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp)) +
      fp * (c2 - c3 * dro_dg * roe * (ro1 * ro1));

  // rhs[FP_ID] = fpp;
  offset = FP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = dfpp_dfpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  offset = F_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[G_ID] = gp;
  offset = G_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0.;
  matrix_data[offset + GP_ID] = dgp_dgp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + G_ID] = dgp_dg;
}

template <std::size_t CTIME_RANK>
void compute_full_rhs_jacobian_cpg(
    const Generic::Vector<double, CTIME_RANK> &state, int state_offset,
    const vector<double> &field, int field_offset,
    Generic::Vector<double, CTIME_RANK> &matrix_data,
    const ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

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

  double fpp = state[state_offset + FPP_ID] / romu;
  double dfpp_dfpp = 1. / romu;
  double dfpp_dg = -dromu_dg * fpp / romu;

  double f = state[state_offset + F_ID];

  double gp = state[state_offset + GP_ID] / romu * prandtl;
  double dgp_dgp = prandtl / romu;
  double dgp_dg = state[state_offset + GP_ID] *
                  (dprandtl_dg / romu - dromu_dg * prandtl / (romu * romu));

  double c1 = params.c1; // 2. * (xi / ue) * due_dxi;
  double c2 = params.c2; // 2. * xi * dhe_dxi / he;
  double c3 = params.c3; // 2. * xi * ue * due_dxi / he;

  // momemtum equation coefficients
  double m0 = field[field_offset + FIELD_M0_ID];
  double m1 = field[field_offset + FIELD_M1_ID];
  double s0 = field[field_offset + FIELD_S0_ID];
  double s1 = field[field_offset + FIELD_S1_ID];

  // energy equation coefficients
  double e0 = field[field_offset + FIELD_E0_ID];
  double e1 = field[field_offset + FIELD_E1_ID];

  int mat_offset;

  // rhs[FPP_ID] =
  //     -f * fpp +
  //     c1 * (fp * fp - roe / ro) +
  //     fp * (m0 * fp + m1) - fpp * (s0 * f + s1);
  mat_offset = FPP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = -f * dfpp_dfpp - (s0 * f + s1) * dfpp_dfpp;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + FP_ID] = 2. * c1 * fp + (2. * m0 * fp + m1);
  matrix_data[mat_offset + F_ID] = -fpp - s0 * fpp;
  matrix_data[mat_offset + G_ID] =
      -f * dfpp_dg + c1 * (roe * dro_dg / (ro * ro)) - (s0 * f + s1) * dfpp_dg;

  // rhs[FP_ID] = fpp;
  mat_offset = FP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = dfpp_dfpp;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + FP_ID] = 0.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  mat_offset = F_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 0;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + FP_ID] = 1.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + G_ID] = 0.;

  // rhs[GP_ID] =
  //   -(f * gp + romu * eckert * fpp * fpp) +
  //   fp * (c2 * g + c3 * (roe / ro)) +
  //   fp * (e0 * g + e1) - gp * (s0 * f + s1);
  mat_offset = GP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[mat_offset + GP_ID] = -f * dgp_dgp - dgp_dgp * (s0 * f * s1);
  matrix_data[mat_offset + FP_ID] = (c2 * g + c3 * (roe / ro)) + (e0 * g + e1);
  matrix_data[mat_offset + F_ID] = -gp - s0 * gp;
  matrix_data[mat_offset + G_ID] =
      -(f * dgp_dg +
        eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp)) +
      fp * (c2 - c3 * (roe * dro_dg) / (ro * ro)) + e0 * fp -
      dgp_dg * (s0 * f + s1);

  // rhs[G_ID] = gp;
  mat_offset = G_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = dgp_dgp;
  matrix_data[mat_offset + FP_ID] = 0.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + G_ID] = dgp_dg;
}

/////
// Output functions
//

inline void compute_outputs_cpg(const vector<double> &state_grid,
                                const vector<double> &eta_grid,
                                vector<double> &output_grid,
                                size_t profile_size,
                                const ProfileParams &profile_params) {
  assert(output_grid.size() >= profile_size * OUTPUT_RANK);

  double delta_eta = profile_params.max_step;

  double pe = profile_params.pe;
  double he = profile_params.he;
  double ue = profile_params.ue;

  double roe = profile_params.roe;
  double mue = profile_params.mue;

  // ro, temperature, cp = thermo_fun(pe, he * g)
  // mu, k = transport_fun(temperature)
  double delta_y;
  double y_old = 0;
  double ro_old = 0;

  int output_offset = 0;
  int state_offset = 0;
  for (int eta_id = 0; eta_id < profile_size; eta_id++) {
    //
    double fp = state_grid[state_offset + FP_ID];
    double g = state_grid[state_offset + G_ID];

    double ro = AIR_CPG_RO(g * he, pe);
    double cp = AIR_CPG_CP(g * he, pe);
    double temperature = pe / (ro * R_AIR);

    double mu = AIR_VISC(temperature);
    double k = AIR_COND(temperature);

    double romu = (ro * mu) / (roe * mue);
    double prandtl = mu * cp / k;

    //
    output_grid[output_offset + OUTPUT_TAU_ID] =
        state_grid[state_offset + FPP_ID];
    output_grid[output_offset + OUTPUT_Q_ID] = state_grid[state_offset + GP_ID];
    output_grid[output_offset + OUTPUT_RO_ID] = roe / g;

    output_grid[output_offset + OUTPUT_MU_ID] = mu;
    output_grid[output_offset + OUTPUT_CHAPMANN_ID] = romu;
    output_grid[output_offset + OUTPUT_PRANDTL_ID] = prandtl;

    //
    if (eta_id >= 1) {
      delta_y = delta_eta * (2. / (ro + ro_old));
      output_grid[output_offset + OUTPUT_Y_ID] = y_old + delta_y;

      y_old = output_grid[output_offset + OUTPUT_Y_ID];
    }

    //
    state_offset += BL_RANK;
    output_offset += OUTPUT_RANK;

    ro_old = ro;
  }
}

// Define bundle
static BLModel<0, 2> cpg_model_functions(
    initialize_cpg, initialize_sensitivity_cpg<0, 2>, compute_rhs_cpg<0>,
    compute_lsim_rhs_cpg<0>, compute_full_rhs_cpg<0>,
    compute_rhs_jacobian_cpg<0>, compute_lsim_rhs_jacobian_cpg<0>,
    compute_full_rhs_jacobian_cpg<0>, limit_update_cpg<0>, compute_outputs_cpg);

// Model that writes to arrays
constexpr int CPG_MODEL_RANK = 5;
constexpr int CPG_TARGET_RANK = 2;
static BLModel<CPG_MODEL_RANK, CPG_TARGET_RANK> cpg_model_functions_stack(
    initialize_cpg, initialize_sensitivity_cpg<CPG_MODEL_RANK, CPG_TARGET_RANK>,
    compute_rhs_cpg<CPG_MODEL_RANK>, compute_lsim_rhs_cpg<CPG_MODEL_RANK>,
    compute_full_rhs_cpg<CPG_MODEL_RANK>,
    compute_rhs_jacobian_cpg<CPG_MODEL_RANK>,
    compute_lsim_rhs_jacobian_cpg<CPG_MODEL_RANK>,
    compute_full_rhs_jacobian_cpg<CPG_MODEL_RANK>,
    limit_update_cpg<CPG_MODEL_RANK>, compute_outputs_cpg);

#endif