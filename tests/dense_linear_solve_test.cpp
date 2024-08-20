#include <cassert>
#include <vector>

#include "dense_direct_solver.h"
#include "dense_linalg.h"
#include "testing_utils.h"

#include <iostream>

void TestReadLowerUpper() {
  int xdim = 5, ydim = 5;
  std::vector<double> matrix_data(xdim * ydim, 0.);

  fillWithRandomData(matrix_data, xdim * ydim);

  int lower_size = xdim * (xdim - 1) / 2;
  int upper_size = xdim * (xdim + 1) / 2;

  std::vector<double> lower_data(lower_size, 0.);
  std::vector<double> upper_data(upper_size, 0.);

  ReadLowerUpper(matrix_data, lower_data, upper_data, xdim);

  std::vector<double> correct_lower_data(lower_size, 0.);
  std::vector<double> correct_upper_data(upper_size, 0.);

  correct_upper_data[0] = matrix_data[0];
  correct_upper_data[1] = matrix_data[1];
  correct_upper_data[2] = matrix_data[2];
  correct_upper_data[3] = matrix_data[3];
  correct_upper_data[4] = matrix_data[4];

  correct_lower_data[0] = matrix_data[5];
  correct_upper_data[5] = matrix_data[6];
  correct_upper_data[6] = matrix_data[7];
  correct_upper_data[7] = matrix_data[8];
  correct_upper_data[8] = matrix_data[9];

  correct_lower_data[1] = matrix_data[10];
  correct_lower_data[2] = matrix_data[11];
  correct_upper_data[9] = matrix_data[12];
  correct_upper_data[10] = matrix_data[13];
  correct_upper_data[11] = matrix_data[14];

  correct_lower_data[3] = matrix_data[15];
  correct_lower_data[4] = matrix_data[16];
  correct_lower_data[5] = matrix_data[17];
  correct_upper_data[12] = matrix_data[18];
  correct_upper_data[13] = matrix_data[19];

  correct_lower_data[6] = matrix_data[20];
  correct_lower_data[7] = matrix_data[21];
  correct_lower_data[8] = matrix_data[22];
  correct_lower_data[9] = matrix_data[23];
  correct_upper_data[14] = matrix_data[24];

  assert(allClose(lower_data, correct_lower_data, lower_size));
  assert(allClose(upper_data, correct_upper_data, upper_size));
}

void TestLowerSolve() {
  int xdim = 5;
  int lower_size = xdim * (xdim - 1) / 2;

  // Trivial case of the unit diagonal
  std::vector<double> unit_lower_data(lower_size, 0.);

  std::vector<double> rhs(xdim, 0.);
  fillWithRandomData(rhs, xdim);

  std::vector<double> solution = rhs;
  LowerSolve(unit_lower_data, rhs, xdim);

  assert(allClose(solution, rhs, xdim));

  // 5 by 5 case
  std::vector<double> lower_data(lower_size, 0.);
  fillWithRandomData(lower_data, lower_size);

  solution[0] = rhs[0];
  solution[1] = rhs[1] - lower_data[0] * solution[0];
  solution[2] =
      rhs[2] - lower_data[2] * solution[1] - lower_data[1] * solution[0];
  solution[3] = rhs[3] - lower_data[5] * solution[2] -
                lower_data[4] * solution[1] - lower_data[3] * solution[0];
  solution[4] = rhs[4] - lower_data[9] * solution[3] -
                lower_data[8] * solution[2] - lower_data[7] * solution[1] -
                lower_data[6] * solution[0];

  LowerSolve(lower_data, rhs, xdim);
  assert(allClose(solution, rhs, xdim));
}

void TestUpperSolve() {
  int xdim = 5;
  int upper_size = xdim * (xdim + 1) / 2;

  // 5 by 5 case
  std::vector<double> upper_data(upper_size, 0.);
  fillWithRandomData(upper_data, upper_size);

  std::vector<double> rhs(xdim, 0.);
  fillWithRandomData(rhs, xdim);

  std::vector<double> solution(xdim, 0.);

  solution[4] = rhs[4] / upper_data[14];
  solution[3] = (rhs[3] - upper_data[13] * solution[4]) / upper_data[12];
  solution[2] =
      (rhs[2] - upper_data[10] * solution[3] - upper_data[11] * solution[4]) /
      upper_data[9];
  solution[1] = (rhs[1] - upper_data[6] * solution[2] -
                 upper_data[7] * solution[3] - upper_data[8] * solution[4]) /
                upper_data[5];
  solution[0] =
      (rhs[0] - upper_data[1] * solution[1] - upper_data[2] * solution[2] -
       upper_data[3] * solution[3] - upper_data[4] * solution[4]) /
      upper_data[0];

  UpperSolve(upper_data, rhs, xdim);
  assert(allClose(solution, rhs, xdim));
}

void TestLUSolve() {
  int xdim = 5;
  std::vector<double> matrix_data(xdim * xdim, 0.);

  fillWithRandomData(matrix_data, xdim * xdim);

  std::vector<double> rhs(xdim, 0.);
  fillWithRandomData(rhs, xdim);

  std::vector<double> solution = rhs;
  LUSolve(matrix_data, solution, xdim);

  std::vector<double> out(xdim, 0.);
  DenseMatrixMultiply(matrix_data, solution, out, xdim);

  assert(allClose(rhs, out, xdim));
}

void TestLowerMatrixSolve() {
  int xdim = 5;
  int zdim = 3;

  int lower_size = xdim * (xdim - 1) / 2;

  // Problem setup. Random matrix and rhs
  std::vector<double> lower_data(lower_size, 0.);
  fillWithRandomData(lower_data, lower_size);

  std::vector<double> rhs_matrix(xdim * zdim, 0.);
  fillWithRandomData(rhs_matrix, xdim * zdim);

  // Matrix solution
  std::vector<double> matrix_solution = rhs_matrix;
  LowerMatrixSolve(lower_data, matrix_solution, xdim, zdim);

  // Sequential solution (column-by-column solve)
  std::vector<double> seq_solution(xdim * zdim, 0.);
  std::vector<double> solution_buffer(xdim, 0.);

  int cm_offset = 0;
  for (int k = 0; k < zdim; k++) {

    //
    for (int i = 0; i < xdim; i++) {
      solution_buffer[i] = rhs_matrix[cm_offset + i];
    }

    //
    LowerSolve(lower_data, solution_buffer, xdim);

    //
    for (int i = 0; i < xdim; i++) {
      seq_solution[cm_offset + i] = solution_buffer[i];
    }

    //
    cm_offset += xdim;
  }

  assert(allClose(matrix_solution, seq_solution, xdim * zdim));
}

void TestUpperMatrixSolve() {
  int xdim = 5;
  int zdim = 3;

  int upper_size = xdim * (xdim + 1) / 2;

  // Problem setup. Random matrix and rhs
  std::vector<double> upper_data(upper_size, 0.);
  fillWithRandomData(upper_data, upper_size);

  std::vector<double> rhs_matrix(xdim * zdim, 0.);
  fillWithRandomData(rhs_matrix, xdim * zdim);

  // Matrix solution
  std::vector<double> matrix_solution = rhs_matrix;
  UpperMatrixSolve(upper_data, matrix_solution, xdim, zdim);

  // Sequential solution (column-by-column solve)
  std::vector<double> seq_solution(xdim * zdim, 0.);
  std::vector<double> solution_buffer(xdim, 0.);

  int cm_offset = 0;
  for (int k = 0; k < zdim; k++) {

    //
    for (int i = 0; i < xdim; i++) {
      solution_buffer[i] = rhs_matrix[cm_offset + i];
    }

    //
    UpperSolve(upper_data, solution_buffer, xdim);

    //
    for (int i = 0; i < xdim; i++) {
      seq_solution[cm_offset + i] = solution_buffer[i];
    }

    //
    cm_offset += xdim;
  }

  assert(allClose(matrix_solution, seq_solution, xdim * zdim));
}

void TestLUMatrixSolve() {
  int xdim = 5;
  int zdim = 3;

  int mat_size = xdim * xdim;

  // Problem setup. Random matrix and rhs
  std::vector<double> matrix_data(mat_size, 0.);
  fillWithRandomData(matrix_data, mat_size);

  std::vector<double> rhs_matrix(xdim * zdim, 0.);
  fillWithRandomData(rhs_matrix, xdim * zdim);

  // Matrix solution
  std::vector<double> matrix_solution = rhs_matrix;
  LUMatrixSolve(matrix_data, matrix_solution, xdim, zdim);

  // Sequential solution (column-by-column solve)
  std::vector<double> seq_solution(xdim * zdim, 0.);
  std::vector<double> solution_buffer(xdim, 0.);

  int cm_offset = 0;
  for (int k = 0; k < zdim; k++) {

    //
    for (int i = 0; i < xdim; i++) {
      solution_buffer[i] = rhs_matrix[cm_offset + i];
    }

    //
    LUSolve(matrix_data, solution_buffer, xdim);

    //
    for (int i = 0; i < xdim; i++) {
      seq_solution[cm_offset + i] = solution_buffer[i];
    }

    //
    cm_offset += xdim;
  }

  assert(allClose(matrix_solution, seq_solution, xdim * zdim));

  //
  std::vector<double> matrix_out(xdim * zdim, 0.);
  DenseMatrixMatrixMultiply(matrix_data, matrix_solution, matrix_out, xdim,
                            zdim);

  assert(allClose(rhs_matrix, matrix_out, xdim * zdim));
}

int main(int argc, char *argv[]) {
  TestReadLowerUpper();
  TestLowerSolve();
  TestUpperSolve();
  TestLUSolve();

  TestLowerMatrixSolve();
  TestUpperMatrixSolve();
  TestLUMatrixSolve();
}