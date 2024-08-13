#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include <cmath>
#include <iostream>
#include <vector>

#include "dense_matrix.h"

struct NewtonParams {
  double rtol = 1e-6;
  int max_iter = 1000;
  int max_ls_iter = 10;
  bool verbose = false;
};

double vector_norm(const std::vector<double> &x) {
  double out = 0.;
  for (int idx = 0; idx < x.size(); idx++) {
    out += x[idx] * x[idx];
  }
  return sqrt(out);
}

template <typename ObjectiveFun, typename JacobianFun, typename LimitUpdateFun>
bool NewtonSolveDirect(std::vector<double> &initial_guess,
                       ObjectiveFun objective_fun, JacobianFun jacobian_fun,
                       LimitUpdateFun limit_update_fun,
                       const NewtonParams &newton_params) {
  double rtol = newton_params.rtol;
  int max_iter = newton_params.max_iter;
  int max_ls_iter = newton_params.max_ls_iter;
  bool verbose = newton_params.verbose;

  int system_size = initial_guess.size();
  assert(system_size > 0);

  std::vector<double> &state = initial_guess;
  std::vector<double> residual(system_size, 0.);
  std::vector<double> state_varn(system_size, 0.);

  DenseMatrix jacobian_matrix(system_size, system_size);

  objective_fun(state, residual);
  jacobian_fun(state, jacobian_matrix);

  double res_norm = vector_norm(residual);

  if (verbose) {
    printf("\n\n*************************************\n");
    printf("** NEWTON START: res_norm=%.2e **\n", res_norm);
  }

  int iter = 0;
  while (iter < max_iter) {

    // Solve linear system
    jacobian_matrix.Solve(residual, state_varn);

    for (int idx = 0; idx < system_size; idx++) {
      state_varn[idx] *= -1;
    };

    // Line Search
    double alpha = limit_update_fun(state, state_varn);
    if (alpha == 0) {
      printf("** ERROR : Initial line search coeff is zero.\n");
      break;
    }

    for (int idx = 0; idx < system_size; idx++) {
      state[idx] += alpha * state_varn[idx];
    }

    bool success = false;
    for (int ls_iter = 0; ls_iter < max_ls_iter; ls_iter++) {

      objective_fun(state, residual);

      double new_res_norm = vector_norm(residual);
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

    if (verbose)
      printf("**  NEWTON Iter#%d, ||x|| = %.2e, ||dx|| = %.2e, a = "
             "%.2e, ||R|| = %.2e\n",
             iter + 1, vector_norm(state), vector_norm(state_varn), alpha,
             res_norm);

    if (res_norm < rtol) {
      if (verbose)
        printf(" => Solution found. ||R|| = %.2e\n", res_norm);
      break;
    }

    iter += 1;
    jacobian_fun(state, jacobian_matrix);
  }

  if (verbose) {
    printf("\n** NEWTON END: ||R|| = %.2e **\n", res_norm);
    printf("**********************************\n\n");
  }

  return (res_norm < rtol);
}

#endif