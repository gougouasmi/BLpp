#ifndef PROFILE_LSIM_H
#define PROFILE_LSIM_H

#include "profile.h"

double compute_lsim_rhs_default(const std::vector<double> &state,
                                std::vector<double> &rhs, int offset,
                                ProfileParams &params);

double compute_lsim_rhs_cpg(const std::vector<double> &state,
                            std::vector<double> &rhs, int offset,
                            ProfileParams &params);

void compute_lsim_rhs_jacobian_default(const std::vector<double> &state,
                                       std::vector<double> &matrix_data,
                                       ProfileParams &params);
void compute_lsim_rhs_jacobian_cpg(const std::vector<double> &state,
                                   std::vector<double> &matrix_data,
                                   ProfileParams &params);

#endif