#ifndef PROFILE_FUNCTIONS_CPG_HPP
#define PROFILE_FUNCTIONS_CPG_HPP

#include "bl_model_struct.hpp"
#include "profile_struct.hpp"

using std::vector;

extern BLModel cpg_model_functions;

void initialize_cpg(ProfileParams &profile_params, vector<double> &state);

void initialize_sensitivity_cpg(ProfileParams &profile_params,
                                vector<double> &state_sensitivity_cm);

double limit_update_cpg(const vector<double> &state,
                        const vector<double> &state_varn,
                        const ProfileParams &params);

double compute_rhs_cpg(const vector<double> &state, int state_offset,
                       const vector<double> &field, int field_offset,
                       vector<double> &rhs, const ProfileParams &params);

double compute_lsim_rhs_cpg(const vector<double> &state, int state_offset,
                            const vector<double> &field, int field_offset,
                            vector<double> &rhs, const ProfileParams &params);

double compute_full_rhs_cpg(const vector<double> &state, int state_offset,
                            const vector<double> &field, int field_offset,
                            vector<double> &rhs, const ProfileParams &params);

void compute_rhs_jacobian_cpg(const vector<double> &state, int state_offset,
                              const vector<double> &field, int field_offset,
                              vector<double> &matrix_data,
                              const ProfileParams &params);

void compute_lsim_rhs_jacobian_cpg(
    const vector<double> &state, int state_offset, const vector<double> &field,
    int field_offset, vector<double> &matrix_data, const ProfileParams &params);

void compute_full_rhs_jacobian_cpg(
    const vector<double> &state, int state_offset, const vector<double> &field,
    int field_offset, vector<double> &matrix_data, const ProfileParams &params);

void compute_outputs_cpg(const vector<double> &state_grid,
                         const vector<double> &eta_grid,
                         vector<double> &output_grid, size_t profile_size,
                         const ProfileParams &profile_params);

#endif