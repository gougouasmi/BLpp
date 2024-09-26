#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using std::array;
using std::pair;
using std::vector;

#include "dense_direct_solver.h"
#include "dense_linalg.h"
#include "timers.h"

////
// Linalg routines on arrays
//

template <std::size_t xdim>
void ReadLowerUpper(const array<double, xdim * xdim> &matrix_data,
                    array<double, xdim *(xdim - 1) / 2> &lower_data,
                    array<double, xdim *(xdim + 1) / 2> &upper_data) {
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

template <std::size_t xdim>
void FactorizeLU(array<double, xdim *(xdim - 1) / 2> &lower_data,
                 array<double, xdim *(xdim + 1) / 2> &upper_data) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t nb_cols = nb_rows;

  int upper_pivot_offset = 0;
  int lower_pivot_offset = 0;

  for (int j = 0; j < nb_cols - 1; j++) {

    // Compute diagonal term
    double pivot1 = 1. / upper_data[upper_pivot_offset];

    int lower_offset = lower_pivot_offset;
    int upper_offset = upper_pivot_offset + (nb_cols - j);

    for (int i = j + 1; i < nb_rows; i++) {

      // Compute L[i,j]
      lower_data[lower_offset] *= pivot1;

      double gauss_coeff = lower_data[lower_offset];

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

template <std::size_t xdim>
void LowerSolve(const array<double, xdim *(xdim - 1) / 2> &lower_data,
                array<double, xdim> &rhs) {
  constexpr size_t nb_rows = xdim;
  int offset = 0;
  for (int i = 0; i < nb_rows; i++) {
    for (int j = 0; j < i; j++) {
      rhs[i] -= lower_data[offset + j] * rhs[j];
    }
    offset += i;
  }
}

template <std::size_t xdim>
void UpperSolve(const array<double, xdim *(xdim + 1) / 2> &upper_data,
                array<double, xdim> &rhs) {
  constexpr size_t nb_rows = xdim;
  constexpr size_t last_row_id = nb_rows - 1;

  int offset = nb_rows * (nb_rows + 1) / 2 - 1;
  for (int i = 0; i < nb_rows; i++) {

    int row_id = last_row_id - i;
    double pivot1 = 1. / upper_data[offset];

    for (int j = 1; j < i + 1; j++) {
      rhs[row_id] -= upper_data[offset + j] * rhs[row_id + j];
    }
    rhs[row_id] *= pivot1;

    offset -= (i + 2);
  }
}

template <std::size_t xdim>
void LUSolve(const array<double, xdim * xdim> &matrix_data,
             array<double, xdim> &rhs,
             pair<array<double, xdim *(xdim - 1) / 2>,
                  array<double, xdim *(xdim + 1) / 2>> &lu_resources) {

  auto &lower_data = lu_resources.first;
  auto &upper_data = lu_resources.second;

  ReadLowerUpper<xdim>(matrix_data, lower_data, upper_data);

  FactorizeLU<xdim>(lower_data, upper_data);

  LowerSolve<xdim>(lower_data, rhs);
  UpperSolve<xdim>(upper_data, rhs);
}

template <std::size_t xdim>
void DenseMatrixMultiply(const array<double, xdim * xdim> &matrix_data_rm,
                         const array<double, xdim> &input_vector,
                         array<double, xdim> &output) {
  int offset = 0;
  for (int i = 0; i < xdim; i++) {
    output[i] = 0;
    for (int j = 0; j < xdim; j++) {
      output[i] += matrix_data_rm[offset + j] * input_vector[j];
    }
    offset += xdim;
  }
}

template <std::size_t xdim>
bool inline allClose(const array<double, xdim> &arr1,
                     const array<double, xdim> &arr2, double rel_tol = 1e-9) {
  double error = 0., arr1_norm = 0., arr2_norm = 0.;
  for (int id = 0; id < xdim; id++) {
    error += pow(arr1[id] - arr2[id], 2.0);
    arr1_norm += pow(arr1[id], 2.0);
    arr2_norm += pow(arr2[id], 2.0);
  }

  arr1_norm = sqrt(arr1_norm);
  arr2_norm = sqrt(arr2_norm);
  error = sqrt(error);

  return error <= rel_tol * std::fmax(arr1_norm, arr2_norm);
}

////
// Helper functions
//

bool inline allClose(const vector<double> &vec1, const vector<double> &vec2,
                     int size, double rel_tol = 1e-9) {
  assert(vec1.size() >= size);
  assert(vec2.size() >= size);

  double error = 0., vec1_norm = 0., vec2_norm = 0.;
  for (int id = 0; id < size; id++) {
    error += pow(vec1[id] - vec2[id], 2.0);
    vec1_norm += pow(vec1[id], 2.0);
    vec2_norm += pow(vec2[id], 2.0);
  }

  vec1_norm = sqrt(vec1_norm);
  vec2_norm = sqrt(vec2_norm);
  error = sqrt(error);

  return error <= rel_tol * std::fmax(vec1_norm, vec2_norm);
}

template <std::size_t N>
void inline fillArrayWithRandomData(array<double, N> &data, int size) {
  assert(size <= N);
  double denom = 1. / static_cast<double>(RAND_MAX);
  for (int i = 0; i < size; ++i) {
    int random_val = rand();
    data[i] = static_cast<double>(random_val) * denom;
  }
}

int main(int argc, char *argv[]) {
  constexpr size_t xdim = 5;
  constexpr size_t mat_size = xdim * xdim;

  constexpr int nb_reps = 10000;

  ////
  // Array setup
  //

  array<double, mat_size> matrix_a_data;
  fillArrayWithRandomData<mat_size>(matrix_a_data, mat_size);

  array<double, xdim> rhs_a;
  fillArrayWithRandomData<xdim>(rhs_a, xdim);

  constexpr size_t lower_size = xdim * (xdim - 1) / 2;
  constexpr size_t upper_size = xdim * (xdim + 1) / 2;

  pair<array<double, lower_size>, array<double, upper_size>> lu_a_resources;

  array<double, xdim> solution_a = rhs_a;
  LUSolve<xdim>(matrix_a_data, solution_a, lu_a_resources);

  array<double, xdim> out_a;
  std::fill(out_a.begin(), out_a.end(), 0.);

  DenseMatrixMultiply<xdim>(matrix_a_data, solution_a, out_a);

  assert(allClose<xdim>(rhs_a, out_a));

  // Define task
  auto array_task = [&matrix_a_data, &solution_a, xdim, &lu_a_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      LUSolve<xdim>(matrix_a_data, solution_a, lu_a_resources);
    }
  };

  ////
  // Vector setup
  //

  vector<double> matrix_v_data(mat_size);
  vector<double> rhs_v(xdim);

  std::copy(matrix_a_data.begin(), matrix_a_data.end(), matrix_v_data.begin());
  std::copy(rhs_a.begin(), rhs_a.end(), rhs_v.begin());

  vector<vector<double>> lu_v_resources = AllocateLUResources(xdim);

  vector<double> solution_v = rhs_v;
  LUSolve(matrix_v_data, solution_v, xdim, lu_v_resources);

  vector<double> out_v(xdim, 0.);
  DenseMatrixMultiply(matrix_v_data, solution_v, out_v, xdim);

  assert(allClose(rhs_v, out_v, xdim));

  // Define task
  auto vector_task = [&matrix_v_data, &solution_v, xdim, &lu_v_resources]() {
    for (int rep = 0; rep < nb_reps; rep++) {
      LUSolve(matrix_v_data, solution_v, xdim, lu_v_resources);
    }
  };

  /////
  // Measure
  //

  auto vector_duration = timeit(vector_task, 10);
  auto array_duration = timeit(array_task, 10);

  std::cout << "Vector task took " << vector_duration << " secs.\n";
  std::cout << "Array task took " << array_duration << " secs.\n";

  return 0;
}