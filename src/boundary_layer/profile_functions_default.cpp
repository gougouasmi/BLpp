#include "profile_functions_default.h"
#include "gas_model.h"

void initialize_default(ProfileParams &profile_params, vector<double> &state) {
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

void initialize_sensitivity_default(ProfileParams &profile_params,
                                    vector<double> &state_sensitivity_cm) {
  double fpp0 = profile_params.fpp0;
  double gp0 = profile_params.gp0;
  double g0 = profile_params.g0;

  double romu0 = 1.;
  double prandtl0 = 1.;

  std::fill(state_sensitivity_cm.begin(), state_sensitivity_cm.end(), 0.);

  switch (profile_params.wall_type) {
  case WallType::Wall:
    // Sensitivities wrt f''(0)
    state_sensitivity_cm[FPP_ID] = romu0;

    // Sensitivities wrt g'(0)
    state_sensitivity_cm[BL_RANK + GP_ID] = romu0 / prandtl0;
    break;
  case WallType::Adiabatic:
    // Sensitivities wrt f''(0)
    state_sensitivity_cm[FPP_ID] = romu0;

    // Sensitivities wrt g(0)
    state_sensitivity_cm[BL_RANK + G_ID] = 1.;
    break;
  }
}

double limit_update_default(const vector<double> &state,
                            const vector<double> &state_varn,
                            ProfileParams &profile_params) {
  double alpha = 1.;

  // Do not let u become negative
  if (state_varn[FP_ID] < 0) {
    alpha = std::min(alpha, 0.2 * state[FP_ID] / (-state_varn[FP_ID]));
  } else {
    alpha = std::min(alpha,
                     0.2 * (1.2 - state[FP_ID]) / (state_varn[FP_ID] + 1e-30));
  }

  alpha =
      std::min(alpha, 0.2 * state[FPP_ID] / fabs(state_varn[FPP_ID] + 1e-30));
  alpha = std::min(alpha, 0.2 * state[G_ID] / fabs(state_varn[G_ID] + 1e-30));

  return alpha;
}

double compute_rhs_default(const vector<double> &state, int state_offset,
                           const vector<double> &field, int field_offset,
                           vector<double> &rhs, ProfileParams &params) {
  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

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

double compute_lsim_rhs_default(const vector<double> &state, int state_offset,
                                const vector<double> &field, int field_offset,
                                vector<double> &rhs, ProfileParams &params) {
  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

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

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

double compute_full_rhs_default(const vector<double> &state, int state_offset,
                                const vector<double> &field, int field_offset,
                                vector<double> &rhs, ProfileParams &params) {
  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  double xi = params.xi;
  double ue = params.ue;
  double he = params.he;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

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
      -f * fpp + 2. * (xi / ue) * (fp * fp - g) * due_dxi +
      2. * xi * (m00 * fp * fp + m01 * fp + m10 * f * fpp + m11 * fpp);
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] =
      -(f * gp + romu * eckert * fpp * fpp) +
      2. * xi * (fp * g * dhe_dxi / he + g * (ue / he) * fp * due_dxi) +
      2. * xi * (e00 * fp * g + e01 * fp + e10 * gp * f + e11 * gp);
  rhs[G_ID] = gp;

  double limit_step = 0.2 * state[state_offset + G_ID] / abs(rhs[G_ID] + 1e-20);

  return limit_step;
}

void compute_rhs_jacobian_default(const vector<double> &state, int state_offset,
                                  const vector<double> &field, int field_offset,
                                  vector<double> &matrix_data,
                                  ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  int offset;

  // rhs[FPP_ID] = - f * fpp
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f / romu;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[FP_ID] = fpp
  offset = FP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 1. / romu;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[F_ID] = fp
  offset = F_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp)
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * prandtl / romu;
  matrix_data[offset + G_ID] = 0.;

  // rhs[G_ID] = gp;
  offset = G_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = prandtl / romu;
  matrix_data[offset + G_ID] = 0.;
}

void compute_lsim_rhs_jacobian_default(
    const vector<double> &state, int state_offset, const vector<double> &field,
    int field_offset, vector<double> &matrix_data, ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  double xi = params.xi;
  double ue = params.ue;
  double he = params.he;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

  int offset;

  // rhs[FPP_ID] = - f * fpp + 2. * (xi / ue) * (fp * fp - g) * due_dxi
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f / romu;
  matrix_data[offset + FP_ID] = 4. * (xi / ue) * fp * due_dxi;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = -2. * (xi / ue) * due_dxi;

  // rhs[FP_ID] = fpp
  offset = FP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 1. / romu;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[F_ID] = fp
  offset = F_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 1.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = 0.;

  // rhs[GP_ID] =
  //    -(f * gp + romu * eckert * fpp * fpp) +
  //    2. * xi * (fp * g) * (dhe_dxi / he + (ue / he) * due_dxi)
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[offset + FP_ID] =
      2. * xi * g * (dhe_dxi / he + (ue / he) * due_dxi);
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * prandtl / romu;
  matrix_data[offset + G_ID] =
      2. * xi * fp * (dhe_dxi / he + (ue / he) * due_dxi);

  // rhs[G_ID] = gp;
  offset = G_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = 0;
  matrix_data[offset + FP_ID] = 0.;
  matrix_data[offset + F_ID] = 0.;
  matrix_data[offset + GP_ID] = prandtl / romu;
  matrix_data[offset + G_ID] = 0.;
}

void compute_full_rhs_jacobian_default(
    const vector<double> &state, int state_offset, const vector<double> &field,
    int field_offset, vector<double> &matrix_data, ProfileParams &params) {
  assert(matrix_data.size() == BL_RANK * BL_RANK);

  double romu = 1.0;
  double prandtl = 1.0;
  double eckert = 1.0;

  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double fpp = state[state_offset + FPP_ID] / romu;
  double f = state[state_offset + F_ID];
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  double xi = params.xi;
  double ue = params.ue;
  double he = params.he;

  double due_dxi = params.due_dxi;
  double dhe_dxi = params.dhe_dxi;

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
  //   - f * fpp +
  //   2. * (xi / ue) * (fp * fp - g) * due_dxi +
  //   2. * xi * (m00 * fp * fp + m01 * fp + (m10 * f + m11) * fpp);
  mat_offset = FPP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] =
      -f / romu + 2. * xi * (m10 * f + m11) / romu;
  matrix_data[mat_offset + FP_ID] =
      4. * (xi / ue) * fp * due_dxi + 2. * xi * (2. * m00 * fp + m01);
  matrix_data[mat_offset + F_ID] = -fpp + 2. * xi * m10 * fpp;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] = -2. * (xi / ue) * due_dxi;

  // rhs[FP_ID] = fpp
  mat_offset = FP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 1. / romu;
  matrix_data[mat_offset + FP_ID] = 0.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] = 0.;

  // rhs[F_ID] = fp
  mat_offset = F_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 0;
  matrix_data[mat_offset + FP_ID] = 1.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] = 0.;

  // rhs[GP_ID] =
  //    -(f * gp + romu * eckert * fpp * fpp) +
  //    2. * xi * (fp * g) * (dhe_dxi / he + (ue / he) * due_dxi) +
  //    2. * xi * ((e00 * g + e01) * fp + e10 * gp * f + e11 * gp);
  mat_offset = GP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[mat_offset + FP_ID] =
      2. * xi * g * (dhe_dxi / he + (ue / he) * due_dxi) +
      2. * xi * (e00 * g + e01);
  matrix_data[mat_offset + F_ID] = -gp + 2. * xi * e10 * gp;
  matrix_data[mat_offset + GP_ID] =
      -f * prandtl / romu + 2. * xi * (e10 * f + e11) * prandtl / romu;
  matrix_data[mat_offset + G_ID] =
      2. * xi * fp * (dhe_dxi / he + (ue / he) * due_dxi) + 2. * xi * e00 * fp;

  // rhs[G_ID] = gp;
  mat_offset = G_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = 0;
  matrix_data[mat_offset + FP_ID] = 0.;
  matrix_data[mat_offset + F_ID] = 0.;
  matrix_data[mat_offset + GP_ID] = prandtl / romu;
  matrix_data[mat_offset + G_ID] = 0.;
}

/////
// Output functions
//

void compute_outputs_default(const vector<double> &state_grid,
                             const vector<double> &eta_grid,
                             vector<double> &output_grid, size_t profile_size,
                             const ProfileParams &profile_params) {
  assert(output_grid.size() >= profile_size * OUTPUT_RANK);

  int output_offset = 0;
  int state_offset = 0;
  for (int eta_id = 0; eta_id < profile_size; eta_id++) {
    //
    double fp = state_grid[state_offset + FP_ID];
    double g = state_grid[state_offset + G_ID];

    double romu = 1.;
    double prandtl = 1.;

    //
    output_grid[output_offset + OUTPUT_U_ID] = fp;
    output_grid[output_offset + OUTPUT_H_ID] = g;
    output_grid[output_offset + OUTPUT_RO_ID] = 1. / g;

    output_grid[output_offset + OUTPUT_CHAPMANN_ID] = romu;
    output_grid[output_offset + OUTPUT_PRANDTL_ID] = prandtl;

    //
    state_offset += BL_RANK;
    output_offset += OUTPUT_RANK;
  }
}

// Define bundle
BLModel default_model_functions(initialize_default,
                                initialize_sensitivity_default,
                                compute_rhs_default, compute_lsim_rhs_default,
                                compute_full_rhs_default,
                                compute_rhs_jacobian_default,
                                compute_lsim_rhs_jacobian_default,
                                compute_full_rhs_jacobian_default,
                                limit_update_default, compute_outputs_default);