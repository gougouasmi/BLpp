#include "profile_lsim.h"
#include "gas_model.h"

double compute_lsim_rhs_default(const std::vector<double> &state,
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

  double xi = params.xi;
  double ue = params.ue;
  double he = params.he;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

  rhs[FPP_ID] = -f * fpp + 2. * (xi / ue) * (fp * fp - g) * due_dxi;
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp) +
               2. * xi * (fp * g * dhe_dxi / he + g * (ue / he) * fp * due_dxi);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

void compute_lsim_rhs_jacobian_default(const std::vector<double> &state,
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

  double xi = params.xi;
  double ue = params.ue;
  double he = params.he;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

  int offset;

  // rhs[FPP_ID] = - f * fpp + 2. * (xi / ue) * (fp * fp - g) * due_dxi
  offset = FPP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -f / romu;
  matrix_data[offset + FP_ID] = 4. * (xi / ue) * fp * due_dxi;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = -2. * (xi / ue) * due_dxi;

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

  // rhs[GP_ID] =
  //    -(f * gp + romu * eckert * fpp * fpp) +
  //    2. * xi * (fp * g) * (dhe_dxi / he + (ue / he) * due_dxi)
  offset = GP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[offset + FP_ID] =
      2. * xi * g * (dhe_dxi / he + (ue / he) * due_dxi);
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * prandtl / romu;
  matrix_data[offset + G_ID] =
      2. * xi * fp * (dhe_dxi / he + (ue / he) * due_dxi);

  // rhs[G_ID] = gp;
  offset = G_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = prandtl / romu;
  matrix_data[offset + G_ID] = 0.;
}

double compute_lsim_rhs_cpg(const std::vector<double> &state,
                            std::vector<double> &rhs, int offset,
                            ProfileParams &params) {
  //
  double fp = state[offset + FP_ID];
  double g = state[offset + G_ID];

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

  double fpp = state[offset + FPP_ID] / romu;
  double f = state[offset + F_ID];
  double gp = state[offset + GP_ID] / romu * prandtl;

  rhs[FPP_ID] = -f * fpp + 2. * (xi / ue) * (fp * fp - roe / ro) * due_dxi;
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] =
      -(f * gp + romu * eckert * fpp * fpp) +
      2. * xi * (fp * g * dhe_dxi / he + (roe * ue) / (ro * he) * fp * due_dxi);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

void compute_lsim_rhs_jacobian_cpg(const std::vector<double> &state,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params) {
  assert(matrix_data.size() == FLAT_PLATE_RANK * FLAT_PLATE_RANK);

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
  offset = FPP_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = -f * dfpp_dfpp;
  matrix_data[offset + FP_ID] = 4. * (xi / ue) * fp * due_dxi;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] =
      -f * dfpp_dg + 2. * (xi / ue) * (roe * dro_dg / (ro * ro)) * due_dxi;

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

  // rhs[GP_ID] =
  //   -(f * gp + romu * eckert * fpp * fpp) +
  //   2. * xi * fp * (g * dhe_dxi / he + (roe * ue) / (ro * he) * due_dxi);
  //
  offset = GP_ID * FLAT_PLATE_RANK;
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
  offset = G_ID * FLAT_PLATE_RANK;
  matrix_data[offset + FPP_ID] = 0.;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = dgp_dgp;
  matrix_data[offset + G_ID] = dgp_dg;
}