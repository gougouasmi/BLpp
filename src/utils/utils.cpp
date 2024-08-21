#include "utils.h"
#include <cassert>

void print_state(std::vector<double> &state, int offset) {
  printf("%f %f %f %f %f\n", state[offset], state[offset + 1],
         state[offset + 2], state[offset + 3], state[offset + 4]);
}

void print_matrix_column_major(std::vector<double> &matrix_data, int xdim,
                               int ydim) {
  assert(matrix_data.size() >= xdim * ydim);

  printf("[\n");
  for (int row_id = 0; row_id < xdim; row_id++) {
    printf(" [");
    int offset = row_id;
    for (int col_id = 0; col_id < ydim; col_id++) {
      printf("%.3e, ", matrix_data[offset]);
      offset += xdim;
    }
    printf("],\n");
  }
  printf("].\n");
}

void print_matrix_row_major(std::vector<double> &matrix_data, int xdim,
                            int ydim) {
  assert(matrix_data.size() >= xdim * ydim);

  int offset = 0;
  printf("[\n");
  for (int row_id = 0; row_id < xdim; row_id++) {
    printf(" [");
    for (int col_id = 0; col_id < ydim; col_id++) {
      printf("%.3e, ", matrix_data[offset + col_id]);
    }
    printf("],\n");
    offset += ydim;
  }
  printf("].\n");
}