#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>

namespace utils {
void print_vector(std::vector<double> &state, int offset, int state_rank,
                  size_t left_space = 0);
void print_matrix_column_major(std::vector<double> &matrix_data, int xdim,
                               int ydim, size_t left_space = 0);
void print_matrix_row_major(std::vector<double> &matrix_data, int xdim,
                            int ydim, size_t left_space = 0);
} // namespace utils

#endif