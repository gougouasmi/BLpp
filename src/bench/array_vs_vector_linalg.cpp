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
#include "dense_linalg.hpp"
#include "timers.hpp"

////
// Linalg routines on arrays
//

template <typename T, std::size_t xdim>
void ReadLowerUpper(const array<T, xdim * xdim> &matrix_data,
                    array<T, xdim *(xdim - 1) / 2> &lower_data,
                    array<T, xdim *(xdim + 1) / 2> &upper_data) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t nb_cols = nb_rows;

  int upper_offset = 0;
  int lower_offset = 0;

  for (int row_id = 0; row_id < nb_rows; row_id++) {

    for (int col_id = 0; col_id < row_id; col_id++) {
      lower_data[lower_offset + col_id] =
          matrix_data[row_id * nb_cols + col_id];
    }

    for (int col_id = row_id; col_id < nb_cols; col_id++) {
      upper_data[upper_offset + (col_id - row_id)] =
          matrix_data[row_id * nb_cols + col_id];
    }

    upper_offset += (nb_cols - row_id);
    lower_offset += row_id;
  }
}

template <typename T, std::size_t xdim>
void FactorizeLU(array<T, xdim *(xdim - 1) / 2> &lower_data,
                 array<T, xdim *(xdim + 1) / 2> &upper_data) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t nb_cols = nb_rows;

  int upper_pivot_offset = 0;
  int lower_pivot_offset = 0;

  for (int j = 0; j < nb_cols - 1; j++) {

    // Compute diagonal term
    T pivot1 = 1. / upper_data[upper_pivot_offset];

    int lower_offset = lower_pivot_offset;
    int upper_offset = upper_pivot_offset + (nb_cols - j);

    for (int i = j + 1; i < nb_rows; i++) {

      // Compute L[i,j]
      lower_data[lower_offset] *= pivot1;

      T gauss_coeff = lower_data[lower_offset];

      // Compute L[i, j:i]
      for (int jb = j + 1; jb < i; jb++) {
        lower_data[lower_offset + jb - j] -=
            (gauss_coeff * upper_data[upper_pivot_offset + jb - j]);
      }

      // Compute U[i, i:nb_cols]
      for (int jb = i; jb < nb_cols; jb++) {
        upper_data[upper_offset + (jb - i)] -=
            (gauss_coeff * upper_data[upper_pivot_offset + jb - j]);
      }

      // Update offset
      lower_offset += i;
      upper_offset += (nb_cols - i);
    }

    // Update pivot offset
    upper_pivot_offset += (nb_cols - j);
    lower_pivot_offset += (j + 2);
  }
}

template <typename T, std::size_t xdim>
void LowerSolve(const array<T, xdim *(xdim - 1) / 2> &lower_data,
                array<T, xdim> &rhs) {
  constexpr size_t nb_rows = xdim;
  int offset = 0;
  for (int i = 0; i < nb_rows; i++) {
    for (int j = 0; j < i; j++) {
      rhs[i] -= lower_data[offset + j] * rhs[j];
    }
    offset += i;
  }
}

template <typename T, std::size_t xdim>
void UpperSolve(const array<T, xdim *(xdim + 1) / 2> &upper_data,
                array<T, xdim> &rhs) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t last_row_id = nb_rows - 1;

  int offset = nb_rows * (nb_rows + 1) / 2 - 1;
  for (int i = 0; i < nb_rows; i++) {

    int row_id = last_row_id - i;
    T pivot1 = 1. / upper_data[offset];

    for (int j = 1; j < i + 1; j++) {
      rhs[row_id] -= upper_data[offset + j] * rhs[row_id + j];
    }
    rhs[row_id] *= pivot1;

    offset -= (i + 2);
  }
}

template <typename T, std::size_t xdim>
void LUSolve(const array<T, xdim * xdim> &matrix_data, array<T, xdim> &rhs,
             pair<array<T, xdim *(xdim - 1) / 2>,
                  array<T, xdim *(xdim + 1) / 2>> &lu_resources) {

  auto &lower_data = lu_resources.first;
  auto &upper_data = lu_resources.second;

  ReadLowerUpper<T, xdim>(matrix_data, lower_data, upper_data);

  FactorizeLU<T, xdim>(lower_data, upper_data);

  LowerSolve<T, xdim>(lower_data, rhs);
  UpperSolve<T, xdim>(upper_data, rhs);
}

template <typename T, std::size_t xdim>
void DenseMatrixMultiply(const array<T, xdim * xdim> &matrix_data_rm,
                         const array<T, xdim> &input_vector,
                         array<T, xdim> &output) {
  int offset = 0;
  for (int i = 0; i < xdim; i++) {
    output[i] = 0;
    for (int j = 0; j < xdim; j++) {
      output[i] += matrix_data_rm[offset + j] * input_vector[j];
    }
    offset += xdim;
  }
}

template <typename T, std::size_t xdim>
bool inline allClose(const array<T, xdim> &arr1, const array<T, xdim> &arr2,
                     T rel_tol = 1e-9) {
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

  // if (!close_enough)
  //   printf("allClose: relative error %.3e > rtol = %.3e", error / max_error,
  //          rel_tol);

  return close_enough;
}

////
// Helper functions
//

template <typename T>
bool inline allClose(const vector<T> &vec1, const vector<T> &vec2, int size,
                     T rel_tol = 1e-9) {
  assert(vec1.size() >= size);
  assert(vec2.size() >= size);

  T error = 0., vec1_norm = 0., vec2_norm = 0.;
  for (int id = 0; id < size; id++) {
    error += pow(vec1[id] - vec2[id], 2.0);
    vec1_norm += pow(vec1[id], 2.0);
    vec2_norm += pow(vec2[id], 2.0);
  }

  vec1_norm = sqrt(vec1_norm);
  vec2_norm = sqrt(vec2_norm);
  error = sqrt(error);

  T max_error = std::fmax(vec1_norm, vec2_norm);

  bool close_enough = error <= rel_tol * max_error;

  // if (!close_enough)
  //   printf("allClose: relative error %.3e > rtol = %.3e", error / max_error,
  //          rel_tol);

  return close_enough;
}

template <typename T, std::size_t N>
void inline fillArrayWithRandomData(array<T, N> &data, int size) {
  assert(size <= N);
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

  constexpr size_t mat_size = xdim * xdim;

  constexpr int nb_reps = 10000;

  ////
  // Array setup
  //

  array<double, mat_size> matrix_a_data;
  fillArrayWithRandomData(matrix_a_data, mat_size);

  array<double, xdim> rhs_a;
  fillArrayWithRandomData(rhs_a, xdim);

  constexpr size_t lower_size = xdim * (xdim - 1) / 2;
  constexpr size_t upper_size = xdim * (xdim + 1) / 2;

  pair<array<double, lower_size>, array<double, upper_size>> lu_a_resources;

  array<double, xdim> solution_a = rhs_a;
  LUSolve(matrix_a_data, solution_a, lu_a_resources);

  array<double, xdim> out_a;
  std::fill(out_a.begin(), out_a.end(), 0.);

  DenseMatrixMultiply(matrix_a_data, solution_a, out_a);

  assert(allClose(rhs_a, out_a));

  // Define task
  auto array_task = [&matrix_a_data, &solution_a, &lu_a_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      LUSolve(matrix_a_data, solution_a, lu_a_resources);
    }
  };

  ////
  // (Float) Array setup
  //

  array<float, mat_size> float_matrix_a_data;
  std::copy(matrix_a_data.begin(), matrix_a_data.end(),
            float_matrix_a_data.begin());

  array<float, xdim> float_rhs_a;
  std::copy(rhs_a.begin(), rhs_a.end(), float_rhs_a.begin());

  pair<array<float, lower_size>, array<float, upper_size>> float_lu_a_resources;

  array<float, xdim> float_solution_a = float_rhs_a;
  LUSolve(float_matrix_a_data, float_solution_a, float_lu_a_resources);

  array<float, xdim> float_out_a;
  std::fill(float_out_a.begin(), float_out_a.end(), 0.);

  DenseMatrixMultiply(float_matrix_a_data, float_solution_a, float_out_a);

  // IMPORTANT NOTE! Solution accuracy is significantly lower with float
  bool float_solution_is_good_enough =
      allClose<float, xdim>(float_rhs_a, float_out_a);

  // Define task
  auto float_array_task = [&float_matrix_a_data, &float_solution_a,
                           &float_lu_a_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      LUSolve(float_matrix_a_data, float_solution_a, float_lu_a_resources);
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

  assert(allClose(rhs_v, out_v, xdim));

  // Define task
  auto vector_task = [&matrix_v_data, &solution_v, &lu_v_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      LUSolve(matrix_v_data, solution_v, xdim, lu_v_resources);
    }
  };

  /////
  // Measure
  //

  auto vector_duration = timeit(vector_task, 10);
  auto array_duration = timeit(array_task, 10);
  auto float_array_duration = timeit(float_array_task, 10);

  std::cout << "\n--- N = " << xdim << " ---\n";
  std::cout << "Vector task took " << vector_duration << " secs.\n";
  std::cout << "Array task took " << array_duration << " secs.\n";
  std::cout << "Array<float> task took " << float_array_duration
            << " secs.\n\n";
}

int main(int argc, char *argv[]) {
  run_lu_benchmark<2>();
  run_lu_benchmark<5>();
  run_lu_benchmark<10>();
  run_lu_benchmark<15>();
}