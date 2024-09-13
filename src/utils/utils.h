#ifndef UTILS_H
#define UTILS_H

#include <vector>

void print_state(std::vector<double> &state, int offset, int state_rank);
void print_matrix_column_major(std::vector<double> &matrix_data, int xdim,
                               int ydim);
void print_matrix_row_major(std::vector<double> &matrix_data, int xdim,
                            int ydim);

#endif