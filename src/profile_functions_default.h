#ifndef PROFILE_FUNCTIONS_DEFAULT_H
#define PROFILE_FUNCTIONS_DEFAULT_H

#include "profile_struct.h"
#include <vector>

void initialize_default(ProfileParams &profile_params,
                        std::vector<double> &state);

double compute_rhs_default(const std::vector<double> &state, int state_offset,
                           const std::vector<double> &field, int field_offset,
                           std::vector<double> &rhs, ProfileParams &params);

double compute_lsim_rhs_default(const std::vector<double> &state,
                                int state_offset,
                                const std::vector<double> &field,
                                int field_offset, std::vector<double> &rhs,
                                ProfileParams &params);

double compute_full_rhs_default(const std::vector<double> &state,
                                int state_offset,
                                const std::vector<double> &field,
                                int field_offset, std::vector<double> &rhs,
                                ProfileParams &params);

void compute_rhs_jacobian_default(const std::vector<double> &state,
                                  const std::vector<double> &field,
                                  int field_offset,
                                  std::vector<double> &matrix_data,
                                  ProfileParams &params);

void compute_lsim_rhs_jacobian_default(const std::vector<double> &state,
                                       const std::vector<double> &field,
                                       int field_offset,
                                       std::vector<double> &matrix_data,
                                       ProfileParams &params);

void compute_full_rhs_jacobian_default(const std::vector<double> &state,
                                       const std::vector<double> &field,
                                       int field_offset,
                                       std::vector<double> &matrix_data,
                                       ProfileParams &params);

#endif