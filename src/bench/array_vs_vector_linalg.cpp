#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using std::array;
using std::pair;
using std::vector;

#include "dense_direct_solver.hpp"
#include "dense_direct_solver_array.hpp"

#include "dense_linalg.hpp"
#include "dense_linalg_array.hpp"
#include "timers.hpp"

#include "generic_dense_direct_solver.hpp"
#include "generic_vector.hpp"

////
// Linalg routines on arrays
//

template <typename T, std::size_t ctime_xdim = 0>
inline bool allClose(const Generic::Vector<T, ctime_xdim> &arr1,
                     const Generic::Vector<T, ctime_xdim> &arr2,
                     T rel_tol = 1e-9) {
  const size_t xdim = arr1.size();
  assert(xdim == arr2.size());

  T error = 0., arr1_norm = 0., arr2_norm = 0.;
  for (int id = 0; id < xdim; id++) {
    error += pow(arr1[id] - arr2[id], 2.0);
    arr1_norm += pow(arr1[id], 2.0);
    arr2_norm += pow(arr2[id], 2.0);
  }

  arr1_norm = sqrt(arr1_norm);
  arr2_norm = sqrt(arr2_norm);
  error = sqrt(error);

  T max_error = std::fmax(arr1_norm, arr2_norm);

  bool close_enough = error <= rel_tol * max_error;

  return close_enough;
}

template <typename T, std::size_t ctime_xdim = 0>
void inline fillArrayWithRandomData(Generic::Vector<T, ctime_xdim> &data,
                                    int size) {
  assert(size <= data.size());
  T denom = 1. / static_cast<T>(RAND_MAX);
  for (int i = 0; i < size; ++i) {
    int random_val = rand();
    data[i] = static_cast<T>(random_val) * denom;
  }
}

/////
// Templated benchmark function
//

template <std::size_t xdim> void run_lu_benchmark() {
  static_assert(xdim > 1);

  constexpr std::size_t mat_size = xdim * xdim;

  constexpr int nb_reps = 10000;

  ////
  // Array setup
  //

  array<double, mat_size> matrix_a_data;
  fillArrayWithRandomData<double, mat_size>(matrix_a_data, mat_size);

  array<double, xdim> rhs_a;
  fillArrayWithRandomData<double, xdim>(rhs_a, xdim);

  constexpr size_t lower_size = xdim * (xdim - 1) / 2;
  constexpr size_t upper_size = xdim * (xdim + 1) / 2;

  pair<array<double, lower_size>, array<double, upper_size>> lu_a_resources;

  Generic::Vector<double, xdim> solution_a = rhs_a;
  ArraySolver::LUSolve(matrix_a_data, solution_a, lu_a_resources);

  Generic::Vector<double, xdim> out_a;
  std::fill(out_a.begin(), out_a.end(), 0.);

  ArrayLinalg::DenseMatrixMultiply(matrix_a_data, solution_a, out_a);

  bool close_enough = allClose<double, xdim>(rhs_a, out_a);
  assert(close_enough);

  // Define task
  auto array_task = [&matrix_a_data, &solution_a, &lu_a_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      ArraySolver::LUSolve(matrix_a_data, solution_a, lu_a_resources);
    }
  };

  /////
  // Array-from-generic setup
  //

  Generic::Vector<double, mat_size> matrix_ga_data;
  Generic::Vector<double, xdim> rhs_ga;

  std::copy(matrix_a_data.begin(), matrix_a_data.end(), matrix_ga_data.begin());
  std::copy(rhs_a.begin(), rhs_a.end(), rhs_ga.begin());

  pair<Generic::Vector<double, lower_size>, Generic::Vector<double, upper_size>>
      lu_ga_resources;

  Generic::Vector<double, xdim> solution_ga = rhs_ga;
  Generic::LUSolve<double, xdim>(matrix_ga_data, solution_ga, lu_ga_resources,
                                 xdim);

  Generic::Vector<double, xdim> out_ga;
  std::fill(out_ga.begin(), out_ga.end(), 0.);

  ArrayLinalg::DenseMatrixMultiply(matrix_ga_data, solution_ga, out_ga);

  bool ga_close_enough = allClose<double, xdim>(rhs_ga, out_ga);
  assert(ga_close_enough);

  // Define task
  auto gen_array_task = [&matrix_ga_data, &solution_ga, &lu_ga_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      Generic::LUSolve<double, xdim>(matrix_ga_data, solution_ga,
                                     lu_ga_resources, xdim);
    }
  };

  ////
  // Vector setup
  //

  vector<double> matrix_v_data(mat_size);
  vector<double> rhs_v(xdim);

  std::copy(matrix_a_data.begin(), matrix_a_data.end(), matrix_v_data.begin());
  std::copy(rhs_a.begin(), rhs_a.end(), rhs_v.begin());

  pair<vector<double>, vector<double>> lu_v_resources =
      AllocateLUResources(xdim);

  vector<double> solution_v = rhs_v;
  LUSolve(matrix_v_data, solution_v, xdim, lu_v_resources);

  vector<double> out_v(xdim, 0.);
  DenseMatrixMultiply(matrix_v_data, solution_v, out_v, xdim);

  assert(allClose<double>(rhs_v, out_v, xdim));

  // Define task
  auto vector_task = [&matrix_v_data, &solution_v, &lu_v_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      LUSolve(matrix_v_data, solution_v, xdim, lu_v_resources);
    }
  };

  ////
  // Vector-from-generic setup
  //

  Generic::Vector<double> matrix_gv_data(mat_size);
  Generic::Vector<double> rhs_gv(xdim);

  std::copy(matrix_a_data.begin(), matrix_a_data.end(), matrix_gv_data.begin());
  std::copy(rhs_a.begin(), rhs_a.end(), rhs_gv.begin());

  pair<Generic::Vector<double>, Generic::Vector<double>> lu_gv_resources =
      AllocateLUResources(xdim);

  Generic::Vector<double> solution_gv = rhs_gv;
  Generic::LUSolve<double>(matrix_gv_data, solution_gv, lu_gv_resources, xdim);

  Generic::Vector<double> out_gv(xdim, 0.);
  DenseMatrixMultiply(matrix_gv_data, solution_gv, out_gv, xdim);

  assert(allClose<double>(rhs_gv, out_gv, xdim));

  // Define task
  auto gen_vector_task = [&matrix_gv_data, &solution_gv, &lu_gv_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      Generic::LUSolve<double>(matrix_gv_data, solution_gv, lu_gv_resources,
                               xdim);
    }
  };

  ////
  // (Float) Array setup
  //

  // array<float, mat_size> float_matrix_a_data;
  // std::copy(matrix_a_data.begin(), matrix_a_data.end(),
  //           float_matrix_a_data.begin());

  // array<float, xdim> float_rhs_a;
  // std::copy(rhs_a.begin(), rhs_a.end(), float_rhs_a.begin());

  // pair<array<float, lower_size>, array<float, upper_size>>
  // float_lu_a_resources;

  // array<float, xdim> float_solution_a = float_rhs_a;
  // ArraySolver::LUSolve(float_matrix_a_data, float_solution_a,
  //                      float_lu_a_resources);

  // array<float, xdim> float_out_a;
  // std::fill(float_out_a.begin(), float_out_a.end(), 0.);

  // ArrayLinalg::DenseMatrixMultiply(float_matrix_a_data, float_solution_a,
  //                                  float_out_a);

  //// IMPORTANT NOTE! Solution accuracy is significantly lower with float
  // bool float_solution_is_good_enough =
  //     allClose<float, xdim>(float_rhs_a, float_out_a);

  //// Define task
  // auto float_array_task = [&float_matrix_a_data, &float_solution_a,
  //                          &float_lu_a_resources]() {
  //   for (int rep = 0; rep < nb_reps; rep++) {
  //     ArraySolver::LUSolve(float_matrix_a_data, float_solution_a,
  //                          float_lu_a_resources);
  //   }
  // };

  /////
  // Measure
  //

  auto vector_duration = timeit(vector_task, 10);
  auto gen_vector_duration = timeit(gen_vector_task, 10);

  auto array_duration = timeit(array_task, 10);
  auto gen_array_duration = timeit(gen_array_task, 10);

  // auto float_array_duration = timeit(float_array_task, 10);

  std::cout << "\n--- N = " << xdim << " ---\n";
  std::cout << "Vector<double>             task took " << vector_duration
            << " secs.\n";
  std::cout << "Generic::Vector<double>    task took " << gen_vector_duration
            << " secs.\n\n";

  std::cout << "Array<double, N>           task took " << array_duration
            << " secs.\n";
  std::cout << "Generic::Vector<double, N> task took " << gen_array_duration
            << " secs.\n";
  // std::cout << "Array<float, N>          task took " << float_array_duration
  //           << " secs.\n\n";
}

int main(int argc, char *argv[]) {
  run_lu_benchmark<2>();
  run_lu_benchmark<5>();
  run_lu_benchmark<10>();
  run_lu_benchmark<15>();
}