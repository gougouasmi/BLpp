#include <cassert>

#include "dense_direct_solver.hpp"
#include "generic_dense_direct_solver.hpp"
#include "testing_utils.hpp"

#include <iostream>

template <std::size_t ctime_xdim> void TestReadLowerUpper(size_t xdim) {
  assert(xdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  std::vector<double> matrix_data(xdim * xdim, 0.);
  fillWithRandomData(matrix_data, xdim * xdim);

  const size_t lower_size = xdim * (xdim - 1) / 2;
  const size_t upper_size = xdim * (xdim + 1) / 2;

  std::vector<double> correct_lower_data(lower_size, 0.);
  std::vector<double> correct_upper_data(upper_size, 0.);

  ReadLowerUpper(matrix_data, correct_lower_data, correct_upper_data, xdim);

  // Use generic container
  constexpr std::size_t ctime_lower_size = ctime_xdim * (ctime_xdim - 1) / 2;
  constexpr std::size_t ctime_upper_size = ctime_xdim * (ctime_xdim + 1) / 2;

  Generic::Vector<double, ctime_xdim * ctime_xdim> generic_matrix_data;
  Generic::Vector<double, ctime_lower_size> generic_lower_data;
  Generic::Vector<double, ctime_upper_size> generic_upper_data;

  if constexpr (ctime_xdim == 0) {
    generic_matrix_data.resize(xdim * xdim);
    generic_lower_data.resize(xdim * (xdim - 1) / 2);
    generic_upper_data.resize(xdim * (xdim + 1) / 2);
  }

  std::copy(matrix_data.begin(), matrix_data.end(),
            generic_matrix_data.begin());

  Generic::ReadLowerUpper<double, ctime_xdim>(
      generic_matrix_data, generic_lower_data, generic_upper_data, xdim);

  bool close_matrices = Generic::allClose<double, ctime_xdim * ctime_xdim, 0>(
      generic_matrix_data, matrix_data, lower_size);
  assert(close_matrices);

  bool close_lower = Generic::allClose<double, ctime_lower_size, 0>(
      generic_lower_data, correct_lower_data, lower_size);
  assert(close_lower);

  bool close_upper = Generic::allClose<double, ctime_upper_size, 0>(
      generic_upper_data, correct_upper_data, upper_size);
  assert(close_upper);
}

template <std::size_t ctime_xdim> void TestLowerSolve(size_t xdim) {
  assert(xdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  const size_t lower_size = xdim * (xdim - 1) / 2;

  // Reference
  std::vector<double> rhs(xdim, 0.);
  fillWithRandomData(rhs, xdim);

  std::vector<double> lower_data(lower_size, 0.);
  fillWithRandomData(lower_data, lower_size);

  //
  constexpr int ctime_lower_size = ctime_xdim * (ctime_xdim - 1) / 2;

  Generic::Vector<double, ctime_xdim> generic_rhs;
  Generic::Vector<double, ctime_lower_size> generic_lower_data;

  if constexpr (ctime_xdim == 0) {
    generic_rhs.resize(xdim);
    generic_lower_data.resize(lower_size);
  }

  std::copy(rhs.begin(), rhs.end(), generic_rhs.begin());
  std::copy(lower_data.begin(), lower_data.end(), generic_lower_data.begin());

  //
  LowerSolve(lower_data, rhs, xdim);
  Generic::LowerSolve<double, ctime_xdim>(generic_lower_data, generic_rhs,
                                          xdim);

  //
  bool same_solutions =
      Generic::allClose<double, ctime_xdim, 0>(generic_rhs, rhs);
  assert(same_solutions);
}

template <std::size_t ctime_xdim> void TestUpperSolve(size_t xdim) {
  assert(xdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  const size_t upper_size = xdim * (xdim + 1) / 2;

  // Reference
  std::vector<double> rhs(xdim, 0.);
  fillWithRandomData(rhs, xdim);

  std::vector<double> upper_data(upper_size, 0.);
  fillWithRandomData(upper_data, upper_size);

  //
  constexpr int ctime_upper_size = ctime_xdim * (ctime_xdim + 1) / 2;

  Generic::Vector<double, ctime_xdim> generic_rhs;
  Generic::Vector<double, ctime_upper_size> generic_upper_data;

  if constexpr (ctime_xdim == 0) {
    generic_rhs.resize(xdim);
    generic_upper_data.resize(upper_size);
  }

  std::copy(rhs.begin(), rhs.end(), generic_rhs.begin());
  std::copy(upper_data.begin(), upper_data.end(), generic_upper_data.begin());

  //
  UpperSolve(upper_data, rhs, xdim);
  Generic::UpperSolve<double, ctime_xdim>(generic_upper_data, generic_rhs,
                                          xdim);

  //
  bool same_solutions =
      Generic::allClose<double, ctime_xdim, 0>(generic_rhs, rhs);
  assert(same_solutions);
}

template <std::size_t ctime_xdim, std::size_t ctime_zdim>
void TestLowerMatrixSolve(size_t xdim, size_t zdim) {
  assert(xdim > 1);
  assert(zdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  if constexpr (ctime_zdim != 0)
    assert(ctime_zdim == zdim && "Compile-time dimension is inconsistent");

  const size_t lower_size = xdim * (xdim - 1) / 2;

  // Reference
  std::vector<double> rhs(xdim * zdim, 0.);
  fillWithRandomData(rhs, xdim * zdim);

  std::vector<double> lower_data(lower_size, 0.);
  fillWithRandomData(lower_data, lower_size);

  //
  constexpr int ctime_lower_size = ctime_xdim * (ctime_xdim - 1) / 2;

  Generic::Vector<double, ctime_xdim * ctime_zdim> generic_rhs;
  Generic::Vector<double, ctime_lower_size> generic_lower_data;

  if constexpr (ctime_xdim == 0) {
    generic_rhs.resize(xdim * zdim);
    generic_lower_data.resize(lower_size);
  }

  std::copy(rhs.begin(), rhs.end(), generic_rhs.begin());
  std::copy(lower_data.begin(), lower_data.end(), generic_lower_data.begin());

  //
  LowerMatrixSolve(lower_data, rhs, xdim, zdim);
  Generic::LowerMatrixSolve<double, ctime_xdim, ctime_zdim>(
      generic_lower_data, generic_rhs, xdim, zdim);

  //
  bool same_solutions =
      Generic::allClose<double, ctime_xdim * ctime_zdim, 0>(generic_rhs, rhs);
  assert(same_solutions);
}

template <std::size_t ctime_xdim, std::size_t ctime_zdim>
void TestUpperMatrixSolve(size_t xdim, size_t zdim) {
  assert(xdim > 1);
  assert(zdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  if constexpr (ctime_zdim != 0)
    assert(ctime_zdim == zdim && "Compile-time dimension is inconsistent");

  const size_t upper_size = xdim * (xdim + 1) / 2;

  // Reference
  std::vector<double> rhs(xdim * zdim, 0.);
  fillWithRandomData(rhs, xdim * zdim);

  std::vector<double> upper_data(upper_size, 0.);
  fillWithRandomData(upper_data, upper_size);

  //
  constexpr int ctime_upper_size = ctime_xdim * (ctime_xdim + 1) / 2;

  Generic::Vector<double, ctime_xdim * ctime_zdim> generic_rhs;
  Generic::Vector<double, ctime_upper_size> generic_upper_data;

  if constexpr (ctime_xdim == 0) {
    generic_rhs.resize(xdim * zdim);
    generic_upper_data.resize(upper_size);
  }

  std::copy(rhs.begin(), rhs.end(), generic_rhs.begin());
  std::copy(upper_data.begin(), upper_data.end(), generic_upper_data.begin());

  //
  UpperMatrixSolve(upper_data, rhs, xdim, zdim);
  Generic::UpperMatrixSolve<double, ctime_xdim, ctime_zdim>(
      generic_upper_data, generic_rhs, xdim, zdim);

  //
  bool same_solutions =
      Generic::allClose<double, ctime_xdim * ctime_zdim, 0>(generic_rhs, rhs);
  assert(same_solutions);
}

template <size_t ctime_xdim> void TestLUSolve(size_t xdim) {
  assert(xdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  vector<double> matrix_data(xdim * xdim, 0.);
  std::pair<vector<double>, vector<double>> lu_resources =
      AllocateLUResources(xdim);

  fillWithRandomData(matrix_data, xdim * xdim);

  vector<double> rhs(xdim, 0.);
  fillWithRandomData(rhs, xdim);

  //
  Generic::Vector<double, ctime_xdim * ctime_xdim> generic_matrix_data;
  Generic::Vector<double, ctime_xdim> generic_rhs;
  std::pair<Generic::Vector<double, ctime_xdim *(ctime_xdim - 1) / 2>,
            Generic::Vector<double, ctime_xdim *(ctime_xdim + 1) / 2>>
      generic_lu_resources;

  if constexpr (ctime_xdim != 0) {
    assert(ctime_xdim == xdim);
  } else {
    generic_matrix_data.resize(matrix_data.size());
    generic_rhs.resize(rhs.size());
    generic_lu_resources.first.resize(lu_resources.first.size());
    generic_lu_resources.second.resize(lu_resources.second.size());
  }

  std::copy(matrix_data.begin(), matrix_data.end(),
            generic_matrix_data.begin());
  std::copy(rhs.begin(), rhs.end(), generic_rhs.begin());

  LUSolve(matrix_data, rhs, xdim, lu_resources);
  Generic::LUSolve<double, ctime_xdim>(generic_matrix_data, generic_rhs,
                                       generic_lu_resources, xdim);

  bool same_solutions =
      Generic::allClose<double, ctime_xdim, 0>(generic_rhs, rhs);
  assert(same_solutions);
}

template <size_t ctime_xdim, size_t ctime_zdim>
void TestLUMatrixSolve(size_t xdim, size_t zdim) {
  assert(xdim > 1);
  assert(zdim > 1);

  if constexpr (ctime_xdim != 0)
    assert(ctime_xdim == xdim && "Compile-time dimension is inconsistent");

  if constexpr (ctime_zdim != 0)
    assert(ctime_zdim == zdim && "Compile-time dimension is inconsistent");

  static_assert(!(ctime_xdim == 0 && ctime_zdim != 0));

  vector<double> matrix_data(xdim * xdim, 0.);
  std::pair<vector<double>, vector<double>> lu_resources =
      AllocateLUResources(xdim);

  fillWithRandomData(matrix_data, xdim * xdim);

  vector<double> rhs(xdim * zdim, 0.);
  fillWithRandomData(rhs, xdim * zdim);

  //
  Generic::Vector<double, ctime_xdim * ctime_xdim> generic_matrix_data;
  Generic::Vector<double, ctime_xdim * ctime_zdim> generic_rhs;
  std::pair<Generic::Vector<double, ctime_xdim *(ctime_xdim - 1) / 2>,
            Generic::Vector<double, ctime_xdim *(ctime_xdim + 1) / 2>>
      generic_lu_resources;

  if constexpr (ctime_xdim != 0) {
    assert(ctime_xdim == xdim);
  } else {
    generic_matrix_data.resize(matrix_data.size());
    generic_rhs.resize(rhs.size());
    generic_lu_resources.first.resize(lu_resources.first.size());
    generic_lu_resources.second.resize(lu_resources.second.size());
  }

  std::copy(matrix_data.begin(), matrix_data.end(),
            generic_matrix_data.begin());
  std::copy(rhs.begin(), rhs.end(), generic_rhs.begin());

  LUMatrixSolve(matrix_data, rhs, xdim, zdim, lu_resources);
  Generic::LUMatrixSolve<double, ctime_xdim, ctime_zdim>(
      generic_matrix_data, generic_rhs, generic_lu_resources, xdim, zdim);

  bool same_solutions =
      Generic::allClose<double, ctime_xdim * ctime_zdim, 0>(generic_rhs, rhs);
  assert(same_solutions);
}

int main(int argc, char *argv[]) {
  TestReadLowerUpper<0>(5);
  TestReadLowerUpper<5>(5);

  TestLowerSolve<0>(5);
  TestLowerSolve<5>(5);

  TestUpperSolve<0>(5);
  TestUpperSolve<5>(5);

  TestLUSolve<5>(5);
  TestLUSolve<0>(5);

  TestLowerMatrixSolve<0, 0>(5, 3);
  TestLowerMatrixSolve<5, 3>(5, 3);

  TestUpperMatrixSolve<0, 0>(5, 3);
  TestUpperMatrixSolve<5, 3>(5, 3);

  TestLUMatrixSolve<0, 0>(5, 3);
  TestLUMatrixSolve<5, 3>(5, 3);
}