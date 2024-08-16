#include "profile_functions_cpg.h"
#include "gas_model.h"

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

double compute_rhs_cpg(const std::vector<double> &state, int state_offset,
                       const std::vector<double> &field, int field_offset,
                       std::vector<double> &rhs, ProfileParams &params) {
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
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

double compute_lsim_rhs_cpg(const std::vector<double> &state, int state_offset,
                            const std::vector<double> &field, int field_offset,
                            std::vector<double> &rhs, ProfileParams &params) {
  //
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double pe = params.pe;
  double he = params.he;

  double xi = params.xi;
  double ue = params.ue;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

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

  rhs[FPP_ID] = -f * fpp + 2. * (xi / ue) * (fp * fp - roe / ro) * due_dxi;
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] =
      -(f * gp + romu * eckert * fpp * fpp) +
      2. * xi * (fp * g * dhe_dxi / he + (roe * ue) / (ro * he) * fp * due_dxi);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

double compute_full_rhs_cpg(const std::vector<double> &state, int state_offset,
                            const std::vector<double> &field, int field_offset,
                            std::vector<double> &rhs, ProfileParams &params) {
  //
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double pe = params.pe;
  double he = params.he;

  double xi = params.xi;
  double ue = params.ue;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

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

  // momemtum equation coefficients
  double m00 = field[field_offset + 0];
  double m01 = field[field_offset + 1];
  double m10 = field[field_offset + 2];
  double m11 = field[field_offset + 3];

  // energy equation coefficients
  double e00 = field[field_offset + 4];
  double e01 = field[field_offset + 5];
  double e10 = field[field_offset + 6];
  double e11 = field[field_offset + 7];

  rhs[FPP_ID] =
      -f * fpp + 2. * (xi / ue) * (fp * fp - roe / ro) * due_dxi +
      2. * xi * (m00 * fp * fp + m01 * fp + m10 * f * fpp + m11 * fpp);
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] =
      -(f * gp + romu * eckert * fpp * fpp) +
      2. * xi *
          (fp * g * dhe_dxi / he + (roe * ue) / (ro * he) * fp * due_dxi) +
      2. * xi * (e00 * fp * g + e01 * fp + e10 * gp * f + e11 * gp);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

void compute_rhs_jacobian_cpg(const std::vector<double> &state,
                              const std::vector<double> &field,
                              int field_offset,
                              std::vector<double> &matrix_data,
                              ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

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
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f * dfpp_dfpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = -f * dfpp_dg;

  // rhs[FP_ID] = fpp;
  offset = FP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = dfpp_dfpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  offset = F_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp);
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * dgp_dgp;
  matrix_data[offset + G_ID] = -(
      f * dgp_dg + eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp));

  // rhs[G_ID] = gp;
  offset = G_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = dgp_dgp;
  matrix_data[offset + G_ID] = dgp_dg;
}

void compute_lsim_rhs_jacobian_cpg(const std::vector<double> &state,
                                   const std::vector<double> &field,
                                   int field_offset,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double fp = state[FP_ID];
  double g = state[G_ID];

  double pe = params.pe;
  double he = params.he;
  double ue = params.ue;

  double xi = params.xi;
  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

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

  // rhs[FPP_ID] = -f * fpp + 2. * (xi / ue) * (fp * fp - roe / ro) * due_dxi;
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f * dfpp_dfpp;
  matrix_data[offset + FP_ID] = 4. * (xi / ue) * fp * due_dxi;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] =
      -f * dfpp_dg + 2. * (xi / ue) * (roe * dro_dg / (ro * ro)) * due_dxi;

  // rhs[FP_ID] = fpp;
  offset = FP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = dfpp_dfpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  offset = F_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] =
  //   -(f * gp + romu * eckert * fpp * fpp) +
  //   2. * xi * fp * (g * dhe_dxi / he + (roe * ue) / (ro * he) * due_dxi);
  //
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[offset + FP_ID] =
      2. * xi * (g * dhe_dxi / he + (roe * ue) / (ro * he) * due_dxi);
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * dgp_dgp;
  matrix_data[offset + G_ID] =
      -(f * dgp_dg +
        eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp)) +
      2. * xi * fp *
          (dhe_dxi / he - dro_dg * (roe * ue) / (ro * ro * he) * due_dxi);

  // rhs[G_ID] = gp;
  offset = G_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = dgp_dgp;
  matrix_data[offset + G_ID] = dgp_dg;
}

void compute_full_rhs_jacobian_cpg(const std::vector<double> &state,
                                   const std::vector<double> &field,
                                   int field_offset,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double fp = state[FP_ID];
  double g = state[G_ID];

  double pe = params.pe;
  double he = params.he;
  double ue = params.ue;

  double xi = params.xi;
  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

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

  // momemtum equation coefficients
  double m00 = field[field_offset + 0];
  double m01 = field[field_offset + 1];
  double m10 = field[field_offset + 2];
  double m11 = field[field_offset + 3];

  // energy equation coefficients
  double e00 = field[field_offset + 4];
  double e01 = field[field_offset + 5];
  double e10 = field[field_offset + 6];
  double e11 = field[field_offset + 7];

  int mat_offset;

  // rhs[FPP_ID] =
  //     -f * fpp +
  //     2. * (xi / ue) * (fp * fp - roe / ro) * due_dxi +
  //     2. * xi * (m00 * fp * fp + m01 * fp + (m10 * f + m11) * fpp);
  mat_offset = FPP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] =
      -f * dfpp_dfpp + 2. * xi * (m10 * f + m11) * dfpp_dfpp;
  matrix_data[mat_offset + FP_ID] =
      4. * (xi / ue) * fp * due_dxi + 2. * xi * (2. * m00 * fp + m01);
  matrix_data[mat_offset + F_ID] = -fpp + 2. * xi * m10 * fpp;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] =
      -f * dfpp_dg + 2. * (xi / ue) * (roe * dro_dg / (ro * ro)) * due_dxi +
      2. * xi * (m10 * f + m11) * dfpp_dg;

  // rhs[FP_ID] = fpp;
  mat_offset = FP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = dfpp_dfpp;
  matrix_data[mat_offset + FP_ID] = 0.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] = dfpp_dg;

  // rhs[F_ID] = fp;
  mat_offset = F_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 0;
  matrix_data[mat_offset + FP_ID] = 1.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] = 0.;

  // rhs[GP_ID] =
  //   -(f * gp + romu * eckert * fpp * fpp) +
  //   2. * xi * fp * (g * dhe_dxi / he + (roe * ue) / (ro * he) * due_dxi) +
  //   2. * xi * ((e00 * g + e01) * fp + (e10 * f + e11) * gp);
  mat_offset = GP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = -romu * eckert * 2. * dfpp_dfpp * fpp;
  matrix_data[mat_offset + FP_ID] =
      2. * xi * (g * dhe_dxi / he + (roe * ue) / (ro * he) * due_dxi) +
      2. * xi * (e00 * g + e01);
  matrix_data[mat_offset + F_ID] = -gp + 2. * xi * e10 * gp;
  matrix_data[mat_offset + GP_ID] =
      -f * dgp_dgp + 2. * xi * (e10 * f * e11) * dgp_dgp;
  matrix_data[mat_offset + G_ID] =
      -(f * dgp_dg +
        eckert * (dromu_dg * fpp * fpp + 2. * romu * dfpp_dg * fpp)) +
      2. * xi * fp *
          (dhe_dxi / he - dro_dg * (roe * ue) / (ro * ro * he) * due_dxi) +
      2. * xi * (e00 * fp + (e10 * f + e11) * dgp_dg);

  // rhs[G_ID] = gp;
  mat_offset = G_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 0.;
  matrix_data[mat_offset + FP_ID] = 0.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = dgp_dgp;
  matrix_data[mat_offset + G_ID] = dgp_dg;
}