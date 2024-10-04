#include "edge_solvers.h"
#include "newton_solver.h"

#include <algorithm>
#include <cassert>

int ComputeFromPressureBE(const vector<double> &pressure_field,
                          vector<double> &density_field,
                          vector<double> &velocity_field, double gamma_ref) {
  int grid_size = pressure_field.size();
  assert(grid_size > 0);

  assert(grid_size == density_field.size());
  assert(grid_size == velocity_field.size());

  assert(density_field[0] > 0);
  assert(pressure_field[0] > 0);

  vector<double> state_buffer(2, 0.);

  double ue, roe, delta_p;

  auto objective_fun = [&ue, &roe, &delta_p](const vector<double> &state,
                                             vector<double> &residual) {
    double ro1 = state[0];
    double u1 = state[1];

    residual[0] = u1 * u1 * (ro1 - roe) - delta_p;
    residual[1] = u1 * ro1 * (u1 - ue) + delta_p;
  };

  auto limit_update_fun = [](const vector<double> &state,
                             const vector<double> &state_varn) {
    double alpha = 1.;

    // density should stay positive
    if (state_varn[0] < 0) {
      alpha = std::min(alpha, 0.2 * state[0] / (-state_varn[0]));
    }

    // velocity should remain positve
    if (state_varn[1] < 0) {
      alpha = std::min(alpha, 0.2 * state[1] / (-state_varn[1]));
    }

    return alpha;
  };

  auto jacobian_fun = [&ue, &roe](const vector<double> &state,
                                  DenseMatrix &matrix) {
    vector<double> &matrix_data = matrix.GetData();

    double ro1 = state[0];
    double u1 = state[1];

    // residual[0] = u1 * u1 * (ro1 - roe) - delta_p;
    matrix_data[0 + 0] = u1 * u1;
    matrix_data[0 + 1] = 2. * u1 * (ro1 - roe);

    // residual[1] = u1 * ro1 * (u1 - ue) + delta_p;
    matrix_data[2 + 0] = u1 * (u1 - ue);
    matrix_data[2 + 1] = ro1 * (2. * u1 - ue);
  };

  NewtonParams newton_params;
  newton_params.max_iter = 2000;
  newton_params.rtol = 1e-6;
  newton_params.verbose = false;

  NewtonResources newton_resources(2);

  delta_p = pressure_field[1] - pressure_field[0];
  roe = density_field[0];
  ue = velocity_field[0];

  for (int xid = 1; xid < grid_size; xid++) {

    delta_p = pressure_field[xid] - pressure_field[xid - 1];

    ue = velocity_field[xid - 1];
    roe = density_field[xid - 1];

    // Solve nonlinear system
    state_buffer[0] = roe;
    state_buffer[1] = ue == 0 ? sqrt(fabs(delta_p) / roe) : ue;

    bool pass =
        NewtonSolveDirect(state_buffer, objective_fun, jacobian_fun,
                          limit_update_fun, newton_params, newton_resources);

    if (!pass) {
      printf("\nLocal edge solve unsuccessfull at iter #%d, abort.\n", xid);
      return xid;
      break;
    }

    density_field[xid] = state_buffer[0];
    velocity_field[xid] = state_buffer[1];
  }

  return grid_size;
}

int ComputeFromPressureCN(const vector<double> &pressure_field,
                          vector<double> &density_field,
                          vector<double> &velocity_field, double gamma_ref) {
  int grid_size = pressure_field.size();
  assert(grid_size > 0);

  assert(grid_size == density_field.size());
  assert(grid_size == velocity_field.size());

  assert(density_field[0] > 0);
  assert(pressure_field[0] > 0);

  vector<double> state_buffer(2, 0.);

  double ue, roe, delta_p;

  double gam = gamma_ref;
  double max_ue = pow(2. * gam / (gam - 1), 0.5);

  auto objective_fun = [&ue, &roe, &delta_p](const vector<double> &state,
                                             vector<double> &residual) {
    double ro1 = state[0];
    double u1 = state[1];

    residual[0] = 0.5 * (u1 * u1 + ue * ue) * (ro1 - roe) - delta_p;
    residual[1] = 0.5 * (u1 * ro1 + ue * roe) * (u1 - ue) + delta_p;
  };

  auto limit_update_fun = [max_ue](const vector<double> &state,
                                   const vector<double> &state_varn) {
    double alpha = 1.;

    // density should stay positive
    if (state_varn[0] < 0) {
      alpha = std::min(alpha, 0.2 * state[0] / (-state_varn[0]));
    }

    // velocity should remain positve
    if (state_varn[1] < 0) {
      alpha = std::min(alpha, 0.2 * state[1] / (-state_varn[1]));
    }

    if (state_varn[1] > 0) {
      alpha = std::min(alpha, 0.5 * (max_ue - state[1]) / state_varn[1]);
    }

    if (state[1] > max_ue) {
      return 0.;
    }

    return alpha;
  };

  auto jacobian_fun = [&ue, &roe](const vector<double> &state,
                                  DenseMatrix &matrix) {
    vector<double> &matrix_data = matrix.GetData();

    double ro1 = state[0];
    double u1 = state[1];

    // residual[0] = 0.5 * (u1 * u1 + ue * ue) * (ro1 - roe) - delta_p;
    matrix_data[0 + 0] = 0.5 * (u1 * u1 + ue * ue);
    matrix_data[0 + 1] = u1 * (ro1 - roe);

    // residual[1] = 0.5 * (u1 * ro1 + ue * roe) * (u1 - ue) + delta_p;
    matrix_data[2 + 0] = 0.5 * u1 * (u1 - ue);
    matrix_data[2 + 1] = 0.5 * (ro1 * (u1 - ue) + (u1 * ro1 + ue * roe));
  };

  NewtonParams newton_params;
  newton_params.max_iter = 2000;
  newton_params.rtol = 1e-6;
  newton_params.verbose = false;

  NewtonResources newton_resources(2);

  delta_p = pressure_field[1] - pressure_field[0];
  roe = density_field[0];
  ue = velocity_field[0];

  for (int xid = 1; xid < grid_size; xid++) {

    delta_p = pressure_field[xid] - pressure_field[xid - 1];

    ue = velocity_field[xid - 1];
    roe = density_field[xid - 1];

    // Solve nonlinear system
    state_buffer[0] = roe;
    state_buffer[1] = ue == 0 ? sqrt(fabs(delta_p) / roe) : ue;

    bool pass =
        NewtonSolveDirect(state_buffer, objective_fun, jacobian_fun,
                          limit_update_fun, newton_params, newton_resources);

    if (!pass) {
      printf("\nLocal edge solve unsuccessfull at iter #%d, abort.\n", xid);
      return xid;
    }

    density_field[xid] = state_buffer[0];
    velocity_field[xid] = state_buffer[1];
  }

  return grid_size;
}

int ComputeFromPressureConstantDensity(const vector<double> &pressure_field,
                                       vector<double> &density_field,
                                       vector<double> &velocity_field,
                                       double gamma_ref) {
  int grid_size = pressure_field.size();
  assert(grid_size > 0);

  assert(grid_size == density_field.size());
  assert(grid_size == velocity_field.size());

  assert(density_field[0] > 0);
  assert(pressure_field[0] > 0);

  bool scaled_pressure_vals_are_under_one =
      !std::any_of(pressure_field.begin(), pressure_field.end(),
                   [](double pval) { return pval > 1.; });
  assert(scaled_pressure_vals_are_under_one);

  double ue;

  double gam = gamma_ref;
  double max_ue = pow(2. * gam / (gam - 1), 0.5);

  for (int xid = 1; xid < grid_size; xid++) {

    ue = pow(1 - pressure_field[xid], 0.5);
    density_field[xid] = 1.0;
    velocity_field[xid] = ue;

    if (ue > max_ue) {
      printf("\n edge velocity above total enthalpy limit.\n");
      return xid;
    }
  }

  return grid_size;
}