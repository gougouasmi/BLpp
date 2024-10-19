#ifndef NEWTON_SOLVER_HPP
#define NEWTON_SOLVER_HPP

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "dense_matrix.hpp"

using std::array;
using std::vector;

struct NewtonParams {
  double rtol = 1e-6;
  int max_iter = 1000;
  int max_ls_iter = 10;
  bool verbose = false;
};

struct NewtonResources {
  vector<double> residual;
  vector<double> state;
  vector<double> state_varn;
  DenseMatrix matrix;

  NewtonResources() = default;
  NewtonResources(int system_size)
      : residual(system_size, 0.), state(system_size, 0.),
        state_varn(system_size, 0.), matrix(system_size, system_size){};
};

double inline vector_norm(const vector<double> &x) {
  double out = 0.;
  for (int idx = 0; idx < x.size(); idx++) {
    out += x[idx] * x[idx];
  }
  return sqrt(out);
}

template <std::size_t N> double inline array_norm(const array<double, N> &x) {
  double out = 0.;
  for (int idx = 0; idx < x.size(); idx++) {
    out += x[idx] * x[idx];
  }
  return sqrt(out);
}

template <typename ObjectiveFun, typename JacobianFun, typename LimitUpdateFun>
bool NewtonSolveDirect(ObjectiveFun objective_fun, JacobianFun jacobian_fun,
                       LimitUpdateFun limit_update_fun,
                       const NewtonParams &newton_params,
                       NewtonResources &resources) {
  double rtol = newton_params.rtol;
  int max_iter = newton_params.max_iter;
  int max_ls_iter = newton_params.max_ls_iter;
  bool verbose = newton_params.verbose;

  vector<double> &state = resources.state;
  vector<double> &residual = resources.residual;     //(system_size, 0.);
  vector<double> &state_varn = resources.state_varn; //(system_size, 0.);

  int system_size = state.size();
  assert(system_size > 0);

  DenseMatrix &jacobian_matrix = resources.matrix; //(system_size, system_size);

  objective_fun(state, residual);
  jacobian_fun(state, jacobian_matrix);

  double res_norm = vector_norm(residual);
  double jac_norm = vector_norm(jacobian_matrix.GetData());

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

    double magn_varn = vector_norm(state_varn);

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

    jacobian_fun(state, jacobian_matrix);

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
  }

  if (verbose) {
    printf("\n** NEWTON END: ||R|| = %.2e **\n", res_norm);
    printf("**********************************\n\n");
  }

  return (res_norm < rtol);
}

#endif