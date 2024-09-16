#ifndef PROFILE_FUNCTIONS_DEFAULT_H
#define PROFILE_FUNCTIONS_DEFAULT_H

#include "profile_struct.h"
#include <vector>

using std::vector;

void initialize_default(ProfileParams &profile_params, vector<double> &state);

void initialize_sensitivity_default(ProfileParams &profile_params,
                                    vector<double> &state_sensitivity_cm);

double limit_update_default(const vector<double> &state,
                            const vector<double> &state_varn,
                            ProfileParams &params);

double compute_rhs_default(const vector<double> &state, int state_offset,
                           const vector<double> &field, int field_offset,
                           vector<double> &rhs, ProfileParams &params);

double compute_lsim_rhs_default(const vector<double> &state, int state_offset,
                                const vector<double> &field, int field_offset,
                                vector<double> &rhs, ProfileParams &params);

double compute_full_rhs_default(const vector<double> &state, int state_offset,
                                const vector<double> &field, int field_offset,
                                vector<double> &rhs, ProfileParams &params);

void compute_rhs_jacobian_default(const vector<double> &state, int state_offset,
                                  const vector<double> &field, int field_offset,
                                  vector<double> &matrix_data,
                                  ProfileParams &params);

void compute_lsim_rhs_jacobian_default(
    const vector<double> &state, int state_offset, const vector<double> &field,
    int field_offset, vector<double> &matrix_data, ProfileParams &params);

void compute_full_rhs_jacobian_default(
    const vector<double> &state, int state_offset, const vector<double> &field,
    int field_offset, vector<double> &matrix_data, ProfileParams &params);

#endif