#include "profile_functions_cpg.hpp"
#include "profile_functions_default.hpp"
#include "profile_struct.hpp"
#include "testing_utils.hpp"
#include <cassert>

template <typename RhsFun, typename JacobianFun>
void JacobiansAreCorrect(RhsFun rhs_fun, JacobianFun jacobian_fun) {
  ProfileParams profile_params;

  const int matrix_dim = BL_RANK * BL_RANK;

  std::vector<double> state(BL_RANK, 0.);
  fillWithRandomData(state, BL_RANK);

  std::vector<double> zero_field(BL_RANK, 0.);
  std::vector<double> rhs_ref(BL_RANK, 0.);
  std::vector<double> jacobian_ref(matrix_dim, 0.);

  rhs_fun(state, 0, zero_field, 0., rhs_ref, profile_params);
  jacobian_fun(state, 0, zero_field, 0, jacobian_ref, profile_params);

  double rhs_norm = Generic::VectorNorm(rhs_ref);
  double jacobian_norm = Generic::VectorNorm(jacobian_ref);

  assert(rhs_norm > 0);
  assert(jacobian_norm > 0);

  //
  std::vector<double> state_buffer(BL_RANK, 0.);
  std::vector<double> rhs_buffer(BL_RANK, 0.);

  std::vector<double> fd_steps(3, 0.);
  fd_steps[0] = 1e-1;
  fd_steps[1] = 1e-2;
  fd_steps[2] = 1e-3;

  std::vector<double> fd_errors(3, 0.);
  std::vector<double> jacobian_error(matrix_dim, 0.);

  // Loop over step size
  for (int step_id = 0; step_id < 3; step_id++) {

    double eps = fd_steps[step_id];

    // Loop over Jacobian column indices
    for (int col_id = 0; col_id < BL_RANK; col_id++) {

      // Compute rhs evaluated at perturbed state
      for (int row_id = 0; row_id < BL_RANK; row_id++) {
        state_buffer[row_id] = state[row_id];
      }
      state_buffer[col_id] += eps;

      rhs_fun(state_buffer, 0, zero_field, 0, rhs_buffer, profile_params);

      // Finite difference approximation
      for (int row_id = 0; row_id < BL_RANK; row_id++) {
        jacobian_error[row_id * BL_RANK + col_id] =
            (jacobian_ref[row_id * BL_RANK + col_id] -
             (rhs_buffer[row_id] - rhs_ref[row_id]) / eps);
      }
    }

    fd_errors[step_id] = Generic::VectorNorm(jacobian_error);
  }

  printf("\n||R|| = %.3e, ||A|| = %.3e, e1 = %.3e,  e2 = %.3e, e3 = %.3e.\n",
         rhs_norm, jacobian_norm, fd_errors[0], fd_errors[1], fd_errors[2]);

  assert(fd_errors[0] >= fd_errors[1]);
  assert(fd_errors[1] >= fd_errors[2]);
}

/* RHS function for local-similarity metho
 * should match with Self-Similar implementation.
 * when edge gradients are zero.
 */
void LocalSimilarityIsConsistent() {
  ProfileParams profile_params;

  std::vector<double> state(BL_RANK, 0.);
  fillWithRandomData(state, BL_RANK);

  std::vector<double> zero_field(FIELD_RANK, 0.);
  std::vector<double> rhs_self_sim(BL_RANK, 0.);
  std::vector<double> rhs_local_sim(BL_RANK, 0.);

  compute_lsim_rhs_default(state, 0, zero_field, 0, rhs_local_sim,
                           profile_params);
  compute_rhs_default(state, 0, zero_field, 0, rhs_self_sim, profile_params);

  assert(allClose(rhs_self_sim, rhs_local_sim, BL_RANK));
  assert(Generic::VectorNorm(rhs_self_sim) > 0);

  // Jacobian functions should be the same
  const int matrix_dim = BL_RANK * BL_RANK;
  std::vector<double> jacobian_lsim(matrix_dim, 0.);
  std::vector<double> jacobian_self_sim(matrix_dim, 0.);

  compute_lsim_rhs_jacobian_default(state, 0, zero_field, 0, jacobian_lsim,
                                    profile_params);
  compute_rhs_jacobian_default(state, 0, zero_field, 0, jacobian_self_sim,
                               profile_params);

  assert(allClose(jacobian_self_sim, jacobian_lsim, matrix_dim));
  assert(Generic::VectorNorm(jacobian_self_sim) > 0);

  // If we introduce gradients, the rhs terms should differ
  profile_params.xi = 1.5;
  profile_params.due_dxi = 0.9;
  profile_params.dhe_dxi = -0.4;

  compute_lsim_rhs_default(state, 0, zero_field, 0, rhs_local_sim,
                           profile_params);
  compute_rhs_default(state, 0, zero_field, 0, rhs_self_sim, profile_params);

  assert(!allClose(rhs_local_sim, rhs_self_sim, BL_RANK));
  assert(rhs_local_sim[FPP_ID] != rhs_self_sim[FPP_ID]);
  assert(rhs_local_sim[GP_ID] != rhs_self_sim[GP_ID]);

  assert(rhs_local_sim[FP_ID] == rhs_self_sim[FP_ID]);
  assert(rhs_local_sim[F_ID] == rhs_self_sim[F_ID]);
  assert(rhs_local_sim[G_ID] == rhs_self_sim[G_ID]);
}

/* RHS function for Difference-Differential
 * method should match with Local-Similarity
 * implementation.
 */
void DifferenceDifferentialIsConsistent() {

  ProfileParams profile_params;

  profile_params.xi = 1.5;
  profile_params.due_dxi = 0.9;
  profile_params.dhe_dxi = -0.4;

  std::vector<double> state(BL_RANK, 0.);
  fillWithRandomData(state, BL_RANK);

  std::vector<double> zero_field(FIELD_RANK, 0.);
  std::vector<double> rhs_full(BL_RANK, 0.);
  std::vector<double> rhs_local_sim(BL_RANK, 0.);

  compute_lsim_rhs_default(state, 0, zero_field, 0, rhs_local_sim,
                           profile_params);
  compute_full_rhs_default(state, 0, zero_field, 0, rhs_full, profile_params);

  assert(allClose(rhs_full, rhs_local_sim, BL_RANK));
  assert(Generic::VectorNorm(rhs_full) > 0);

  // Jacobian functions should be the same
  const int matrix_dim = BL_RANK * BL_RANK;
  std::vector<double> jacobian_lsim(matrix_dim, 0.);
  std::vector<double> jacobian_full(matrix_dim, 0.);

  compute_lsim_rhs_jacobian_default(state, 0, zero_field, 0, jacobian_lsim,
                                    profile_params);
  compute_full_rhs_jacobian_default(state, 0, zero_field, 0, jacobian_full,
                                    profile_params);

  assert(allClose(jacobian_lsim, jacobian_full, matrix_dim));
  assert(Generic::VectorNorm(jacobian_full) > 0);

  // If we introduce a non-zero field, the rhs terms should differ
  std::vector<double> non_zero_field(FIELD_RANK, 0.);
  fillWithRandomData(non_zero_field, FIELD_RANK);

  compute_lsim_rhs_default(state, 0, non_zero_field, 0, rhs_local_sim,
                           profile_params);
  compute_full_rhs_default(state, 0, non_zero_field, 0, rhs_full,
                           profile_params);

  assert(!allClose(rhs_local_sim, rhs_full, BL_RANK));
  assert(rhs_local_sim[FPP_ID] != rhs_full[FPP_ID]);
  assert(rhs_local_sim[GP_ID] != rhs_full[GP_ID]);

  assert(rhs_local_sim[FP_ID] == rhs_full[FP_ID]);
  assert(rhs_local_sim[F_ID] == rhs_full[F_ID]);
  assert(rhs_local_sim[G_ID] == rhs_full[G_ID]);
}

int main(int argc, char *argv[]) {
  // Jacobian implementations are consistent with Frechet derivatives
  JacobiansAreCorrect(compute_rhs_default<0>, compute_rhs_jacobian_default<0>);
  JacobiansAreCorrect(compute_lsim_rhs_default<0>,
                      compute_lsim_rhs_jacobian_default<0>);
  JacobiansAreCorrect(compute_full_rhs_default<0>,
                      compute_full_rhs_jacobian_default<0>);

  // Consistency between Self-Similar, Locally-Similar, and
  // Difference-Differential
  LocalSimilarityIsConsistent();
  DifferenceDifferentialIsConsistent();
}