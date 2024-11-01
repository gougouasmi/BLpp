#include "utils.hpp"
#include <cassert>

namespace utils {
void print_vector(const std::vector<double> &state, int offset, int state_rank,
                  size_t left_space) {

  auto make_space = [left_space]() {
    for (int space = 0; space < left_space; space++) {
      printf(" ");
    }
  };

  make_space();

  printf("[");
  for (int state_id = 0; state_id < state_rank - 1; state_id++) {
    printf("%.4e, ", state[offset + state_id]);
  }
  printf("%.4e ].\n", state[offset + state_rank - 1]);
}

void print_matrix_column_major(const std::vector<double> &matrix_data, int xdim,
                               int ydim, size_t left_space) {
  assert(matrix_data.size() >= xdim * ydim);

  auto make_space = [left_space]() {
    for (int space = 0; space < left_space; space++) {
      printf(" ");
    }
  };

  make_space();
  printf("[\n");
  for (int row_id = 0; row_id < xdim; row_id++) {
    make_space();
    printf(" [");
    int offset = row_id;
    for (int col_id = 0; col_id < ydim; col_id++) {
      printf("%.3e, ", matrix_data[offset]);
      offset += xdim;
    }
    printf("],\n");
  }
  make_space();
  printf("].\n");
}

void print_matrix_row_major(const std::vector<double> &matrix_data, int xdim,
                            int ydim, size_t left_space) {
  assert(matrix_data.size() >= xdim * ydim);

  int offset = 0;

  auto make_space = [left_space]() {
    for (int space = 0; space < left_space; space++) {
      printf(" ");
    }
  };

  make_space();
  printf("[\n");
  for (int row_id = 0; row_id < xdim; row_id++) {
    make_space();
    printf(" [");
    for (int col_id = 0; col_id < ydim; col_id++) {
      printf("%.3e, ", matrix_data[offset + col_id]);
    }
    printf("],\n");
    offset += ydim;
  }
  make_space();
  printf("].\n");
}

} // namespace utils