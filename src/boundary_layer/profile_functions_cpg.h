#ifndef PROFILE_FUNCTIONS_CPG_H
#define PROFILE_FUNCTIONS_CPG_H

#include "profile_struct.h"

void initialize_cpg(ProfileParams &profile_params, std::vector<double> &state);

void initialize_sensitivity_cpg(ProfileParams &profile_params,
                                std::vector<double> &state_sensitivity_cm);

double compute_rhs_cpg(const std::vector<double> &state, int state_offset,
                       const std::vector<double> &field, int field_offset,
                       std::vector<double> &rhs, ProfileParams &params);

double compute_lsim_rhs_cpg(const std::vector<double> &state, int state_offset,
                            const std::vector<double> &field, int field_offset,
                            std::vector<double> &rhs, ProfileParams &params);

double compute_full_rhs_cpg(const std::vector<double> &state, int state_offset,
                            const std::vector<double> &field, int field_offset,
                            std::vector<double> &rhs, ProfileParams &params);

void compute_rhs_jacobian_cpg(const std::vector<double> &state,
                              int state_offset,
                              const std::vector<double> &field,
                              int field_offset,
                              std::vector<double> &matrix_data,
                              ProfileParams &params);

void compute_lsim_rhs_jacobian_cpg(const std::vector<double> &state,
                                   int state_offset,
                                   const std::vector<double> &field,
                                   int field_offset,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params);

void compute_full_rhs_jacobian_cpg(const std::vector<double> &state,
                                   int state_offset,
                                   const std::vector<double> &field,
                                   int field_offset,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params);

#endif