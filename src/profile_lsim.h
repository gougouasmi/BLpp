#ifndef PROFILE_LSIM_H
#define PROFILE_LSIM_H

#include "profile.h"

double compute_lsim_rhs_default(const std::vector<double> &state,
                                int state_offset,
                                const std::vector<double> &field,
                                int field_offset, std::vector<double> &rhs,
                                ProfileParams &params);

double compute_lsim_rhs_cpg(const std::vector<double> &state, int state_offset,
                            const std::vector<double> &field, int field_offset,
                            std::vector<double> &rhs, ProfileParams &params);

void compute_lsim_rhs_jacobian_default(const std::vector<double> &state,
                                       const std::vector<double> &field,
                                       int field_offset,
                                       std::vector<double> &matrix_data,
                                       ProfileParams &params);
void compute_lsim_rhs_jacobian_cpg(const std::vector<double> &state,
                                   const std::vector<double> &field,
                                   int field_offset,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params);

#endif