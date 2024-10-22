#ifndef GENERIC_NEWTON_SOLVER_HPP
#define GENERIC_NEWTON_SOLVER_HPP

#include <cmath>
#include <iostream>

#include "generic_dense_matrix.hpp"
#include "generic_vector.hpp"
#include "newton_struct.hpp"

namespace Generic {

template <std::size_t ctime_rank = 0> struct NewtonResources {
  Generic::Vector<double, ctime_rank> residual;
  Generic::Vector<double, ctime_rank> state;
  Generic::Vector<double, ctime_rank> state_varn;
  Generic::DenseMatrix<ctime_rank> matrix;

  NewtonResources() = default;
  NewtonResources(int system_size) : matrix(system_size) {
    if constexpr (ctime_rank == 0) {
      residual.resize(system_size);
      state.resize(system_size);
      state_varn.resize(system_size);
    } else {
      assert(system_size == ctime_rank);
    }
  };
};

template <typename ObjectiveFun, typename JacobianFun, typename LimitUpdateFun,
          std::size_t ctime_system_rank = 0>
bool NewtonSolveDirect(ObjectiveFun objective_fun, JacobianFun jacobian_fun,
                       LimitUpdateFun limit_update_fun,
                       const NewtonParams &newton_params,
                       NewtonResources<ctime_system_rank> &resources) {
  double rtol = newton_params.rtol;
  int max_iter = newton_params.max_iter;
  int max_ls_iter = newton_params.max_ls_iter;
  bool verbose = newton_params.verbose;

  Generic::Vector<double, ctime_system_rank> &state = resources.state;
  Generic::Vector<double, ctime_system_rank> &residual = resources.residual;
  Generic::Vector<double, ctime_system_rank> &state_varn = resources.state_varn;

  int system_size = (ctime_system_rank == 0) ? state.size() : ctime_system_rank;
  assert(system_size > 0);

  Generic::DenseMatrix<ctime_system_rank> &jacobian_matrix = resources.matrix;

  objective_fun(state, residual);
  jacobian_fun(state, jacobian_matrix);

  double res_norm = Generic::VectorNorm<ctime_system_rank>(residual);
  double jac_norm = Generic::VectorNorm<ctime_system_rank * ctime_system_rank>(
      jacobian_matrix.GetData());

  if (verbose) {
    printf("\n*************************************\n");
    printf("** NEWTON START: res_norm=%.2e, jac_norm=%.2e **\n", res_norm,
           jac_norm);
  }

  int iter = 0;
  while (iter < max_iter) {

    // Solve linear system
    jacobian_matrix.Solve(residual, state_varn);

    for (int idx = 0; idx < system_size; idx++) {
      state_varn[idx] *= -1;
    };

    double magn_varn = Generic::VectorNorm<ctime_system_rank>(state_varn);

    // Line Search
    double alpha = limit_update_fun(state, state_varn);
    if (alpha == 0) {
      if (verbose)
        printf("** ERROR : Initial line search coeff is zero, "
               "||state_varn||=%.2e.\n",
               magn_varn);
      break;
    }

    for (int idx = 0; idx < system_size; idx++) {
      state[idx] += alpha * state_varn[idx];
    }

    bool success = false;
    for (int ls_iter = 0; ls_iter < max_ls_iter; ls_iter++) {

      objective_fun(state, residual);

      double new_res_norm = Generic::VectorNorm<ctime_system_rank>(residual);
      success = new_res_norm < res_norm;

      if (success) {
        res_norm = new_res_norm;
        break;
      }

      alpha *= 0.5;
      for (int idx = 0; idx < system_size; idx++) {
        state[idx] -= alpha * state_varn[idx];
      }
    }

    // Review results
    if (!success) {
      if (verbose)
        printf(" => Unsuccessful line search.\n");
      break;
    }

    jacobian_fun(state, jacobian_matrix);

    if (verbose)
      printf("**  NEWTON Iter#%d, ||x|| = %.2e, ||dx|| = %.2e, a = "
             "%.2e, ||R|| = %.2e\n",
             iter + 1, Generic::VectorNorm<ctime_system_rank>(state),
             Generic::VectorNorm<ctime_system_rank>(state_varn), alpha,
             res_norm);

    if (res_norm < rtol) {
      if (verbose)
        printf(" => Solution found. ||R|| = %.2e\n", res_norm);
      break;
    }

    iter += 1;
  }

  if (verbose) {
    printf("\n** NEWTON END: ||R|| = %.2e **\n", res_norm);
    printf("**********************************\n\n");
  }

  return (res_norm < rtol);
}

} // namespace Generic

#endif