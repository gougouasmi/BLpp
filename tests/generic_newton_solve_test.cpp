#include <cassert>
#include <vector>

#include "generic_dense_matrix.hpp"
#include "generic_newton_solver.hpp"
#include "generic_vector.hpp"
#include "testing_utils.hpp"

void ScalarSolveHeap() {
  int xdim = 1;

  auto objective_fun = [](const Generic::Vector<double> &x,
                          Generic::Vector<double> &y) { y[0] = x[0] * x[0]; };

  auto limit_update_fun = [](const Generic::Vector<double> &x,
                             const Generic::Vector<double> &dx) { return 1.; };

  auto jacobian_fun = [](const Generic::Vector<double> &x,
                         Generic::DenseMatrix<0> &matrix) {
    Generic::Vector<double> &matrix_data = matrix.GetData();
    matrix_data[0] = 2 * x[0];
  };

  NewtonParams newton_params;
  newton_params.verbose = true;

  Generic::NewtonResources newton_resources(xdim);

  Generic::Vector<double> &solution = newton_resources.state;
  fillWithRandomData(solution, xdim);

  bool pass =
      Generic::NewtonSolveDirect(objective_fun, jacobian_fun, limit_update_fun,
                                 newton_params, newton_resources);

  Generic::Vector<double> residual(xdim, 0.);
  objective_fun(solution, residual);

  assert(Generic::VectorNorm(residual) < newton_params.rtol);
}

void ScalarSolveStack() {
  constexpr int xdim = 1;

  auto objective_fun = [](const Generic::Vector<double, xdim> &x,
                          Generic::Vector<double, xdim> &y) {
    y[0] = x[0] * x[0];
  };

  auto limit_update_fun = [](const Generic::Vector<double, xdim> &x,
                             const Generic::Vector<double, xdim> &dx) {
    return 1.;
  };

  auto jacobian_fun = [](const Generic::Vector<double, xdim> &x,
                         Generic::DenseMatrix<xdim> &matrix) {
    Generic::Vector<double, xdim> &matrix_data = matrix.GetData();
    matrix_data[0] = 2 * x[0];
  };

  NewtonParams newton_params;
  newton_params.verbose = true;

  Generic::NewtonResources<xdim> newton_resources(xdim);

  Generic::Vector<double, xdim> &solution = newton_resources.state;

  Generic::fillWithRandomData<double, xdim>(solution, xdim);

  bool pass =
      Generic::NewtonSolveDirect(objective_fun, jacobian_fun, limit_update_fun,
                                 newton_params, newton_resources);

  printf("residual = %.3e.\n", newton_resources.residual[0]);

  Generic::Vector<double, xdim> residual;
  objective_fun(solution, residual);

  assert(Generic::VectorNorm<xdim>(residual) < newton_params.rtol);
}

void SystemSolveHeap() {
  int xdim = 3;

  auto objective_fun = [](const Generic::Vector<double> &x,
                          Generic::Vector<double> &y) {
    y[0] = x[0] * x[0] - 1;
    y[1] = x[0] * x[1] + 3;
    y[2] = x[0] + x[1] + x[2];
  };

  auto limit_update_fun = [](const Generic::Vector<double> &x,
                             const Generic::Vector<double> &dx) { return 1.; };

  auto jacobian_fun = [](const Generic::Vector<double> &x,
                         Generic::DenseMatrix<0> &matrix) {
    Generic::Vector<double> &matrix_data = matrix.GetData();
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

  Generic::NewtonResources newton_resources(xdim);

  Generic::Vector<double> &solution = newton_resources.state;
  fillWithRandomData(solution, xdim);

  NewtonParams newton_params;
  newton_params.verbose = true;

  bool pass =
      Generic::NewtonSolveDirect(objective_fun, jacobian_fun, limit_update_fun,
                                 newton_params, newton_resources);

  Generic::Vector<double> residual(xdim, 0.);
  objective_fun(solution, residual);

  assert(Generic::VectorNorm(residual) < newton_params.rtol);
}

void SystemSolveStack() {
  constexpr int xdim = 3;

  auto objective_fun = [](const Generic::Vector<double, xdim> &x,
                          Generic::Vector<double, xdim> &y) {
    y[0] = x[0] * x[0] - 1;
    y[1] = x[0] * x[1] + 3;
    y[2] = x[0] + x[1] + x[2];
  };

  auto limit_update_fun = [](const Generic::Vector<double, xdim> &x,
                             const Generic::Vector<double, xdim> &dx) {
    return 1.;
  };

  auto jacobian_fun = [](const Generic::Vector<double, xdim> &x,
                         Generic::DenseMatrix<xdim> &matrix) {
    Generic::Vector<double, xdim *xdim> &matrix_data = matrix.GetData();
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

  Generic::NewtonResources<xdim> newton_resources;

  Generic::Vector<double, xdim> &solution = newton_resources.state;
  Generic::fillWithRandomData<double, xdim>(solution, xdim);

  NewtonParams newton_params;
  newton_params.verbose = true;

  bool pass =
      Generic::NewtonSolveDirect(objective_fun, jacobian_fun, limit_update_fun,
                                 newton_params, newton_resources);

  Generic::Vector<double, xdim> residual;
  objective_fun(solution, residual);

  assert(Generic::VectorNorm<xdim>(residual) < newton_params.rtol);
}

int main(int argc, char *argv[]) {
  ScalarSolveHeap();
  ScalarSolveStack();

  SystemSolveHeap();
  SystemSolveStack();
}