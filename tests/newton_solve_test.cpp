#include <cassert>
#include <vector>

#include "dense_matrix.hpp"
#include "newton_solver.hpp"
#include "testing_utils.hpp"

void ScalarSolve() {
  int xdim = 1;

  auto objective_fun = [](const std::vector<double> &x,
                          std::vector<double> &y) { y[0] = x[0] * x[0]; };

  auto limit_update_fun = [](const std::vector<double> &x,
                             const std::vector<double> &dx) { return 1.; };

  auto jacobian_fun = [](const std::vector<double> &x, DenseMatrix &matrix) {
    std::vector<double> &matrix_data = matrix.GetData();
    matrix_data[0] = 2 * x[0];
  };

  NewtonParams newton_params;
  NewtonResources newton_resources(xdim);

  std::vector<double> &solution = newton_resources.state;
  fillWithRandomData(solution, xdim);

  bool pass = NewtonSolveDirect(objective_fun, jacobian_fun, limit_update_fun,
                                newton_params, newton_resources);

  std::vector<double> residual(xdim, 0.);
  objective_fun(solution, residual);

  assert(Generic::VectorNorm(residual) < newton_params.rtol);
}

void SystemSolve() {
  int xdim = 3;

  auto objective_fun = [](const std::vector<double> &x,
                          std::vector<double> &y) {
    y[0] = x[0] * x[0] - 1;
    y[1] = x[0] * x[1] + 3;
    y[2] = x[0] + x[1] + x[2];
  };

  auto limit_update_fun = [](const std::vector<double> &x,
                             const std::vector<double> &dx) { return 1.; };

  auto jacobian_fun = [](const std::vector<double> &x, DenseMatrix &matrix) {
    std::vector<double> &matrix_data = matrix.GetData();
    matrix_data[0] = 2 * x[0];
    matrix_data[1] = 0.;
    matrix_data[2] = 0.;

    matrix_data[3] = x[1];
    matrix_data[4] = x[0];
    matrix_data[5] = 0;

    matrix_data[6] = 1.;
    matrix_data[7] = 1.;
    matrix_data[8] = 1.;
  };

  NewtonResources newton_resources(xdim);

  vector<double> &solution = newton_resources.state;
  fillWithRandomData(solution, xdim);

  NewtonParams newton_params;
  bool pass = NewtonSolveDirect(objective_fun, jacobian_fun, limit_update_fun,
                                newton_params, newton_resources);

  std::vector<double> residual(xdim, 0.);
  objective_fun(solution, residual);

  assert(Generic::VectorNorm(residual) < newton_params.rtol);
}

int main(int argc, char *argv[]) {
  ScalarSolve();
  SystemSolve();
}