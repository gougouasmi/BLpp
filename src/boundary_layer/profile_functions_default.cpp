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

  double c1 = 2. * (xi / ue) * due_dxi;
  double c2 = 2. * xi * dhe_dxi / he;
  double c3 = 2. * xi * ue * due_dxi / he;

  // momemtum equation coefficients
  double m0 = field[field_offset + FIELD_M0_ID];
  double m1 = field[field_offset + FIELD_M1_ID];
  double s0 = field[field_offset + FIELD_S0_ID];
  double s1 = field[field_offset + FIELD_S1_ID];

  // energy equation coefficients
  double e0 = field[field_offset + FIELD_E0_ID];
  double e1 = field[field_offset + FIELD_E1_ID];

  rhs[FPP_ID] =
      -f * fpp + c1 * (fp * fp - g) + fp * (m0 * fp + m1) - fpp * (s0 * f + s1);
  rhs[FP_ID] = fpp;
  rhs[F_ID] = fp;

  rhs[GP_ID] = -(f * gp + romu * eckert * fpp * fpp) + fp * g * (c2 + c3) +
               fp * (e0 * g + e1) - gp * (s0 * gp * f + s1);
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

  double c1 = 2. * (xi / ue) * due_dxi;
  double c2 = 2. * xi * dhe_dxi / he;
  double c3 = 2. * xi * ue * due_dxi / he;

  int offset;

  // rhs[FPP_ID] = - f * fpp + c1 * (fp * fp - g)
  offset = FPP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -f / romu;
  matrix_data[offset + FP_ID] = 2. * c1 * fp;
  matrix_data[offset + F_ID] = -fpp;
  matrix_data[offset + GP_ID] = 0.;
  matrix_data[offset + G_ID] = -c1;

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
  //    fp * g * (c2 + c3)
  offset = GP_ID * BL_RANK;
  matrix_data[offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[offset + FP_ID] = g * (c2 + c3);
  matrix_data[offset + F_ID] = -gp;
  matrix_data[offset + GP_ID] = -f * prandtl / romu;
  matrix_data[offset + G_ID] = fp * (c2 + c3);

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

  double c1 = 2. * (xi / ue) * due_dxi;
  double c2 = 2. * xi * dhe_dxi / he;
  double c3 = 2. * xi * ue * due_dxi / he;

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
  //   - f * fpp +
  //   c1 * (fp * fp - g) +
  //   fp * (m0 * fp + m1) - fpp * (s0 * f + s1);
  mat_offset = FPP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = -f / romu - (s0 * f + s1) / romu;
  matrix_data[mat_offset + FP_ID] = 2. * c1 * fp + (2. * m0 * fp + m1);
  matrix_data[mat_offset + F_ID] = -fpp - s0 * fpp;
  matrix_data[mat_offset + GP_ID] = 0.;
  matrix_data[mat_offset + G_ID] = -c1;

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
  //    fp * g * (c2 + c3) +
  //    (fp * (e0 * g + e1) - gp * (s0 * f + s1));
  mat_offset = GP_ID * BL_RANK;
  matrix_data[mat_offset + FPP_ID] = -eckert * 2. * fpp;
  matrix_data[mat_offset + FP_ID] = g * (c2 + c3) + (e0 * g + e1);
  matrix_data[mat_offset + F_ID] = -gp - s0 * gp;
  matrix_data[mat_offset + GP_ID] =
      -f * prandtl / romu - (s0 * f + s1) * prandtl / romu;
  matrix_data[mat_offset + G_ID] = fp * (c2 + c3) + e0 * fp;

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
    output_grid[output_offset + OUTPUT_TAU_ID] =
        state_grid[state_offset + FPP_ID];
    output_grid[output_offset + OUTPUT_Q_ID] = state_grid[state_offset + GP_ID];
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