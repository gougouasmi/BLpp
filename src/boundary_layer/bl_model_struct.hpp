#ifndef BL_MODEL_HPP
#define BL_MODEL_HPP

#include "profile_struct.hpp"
#include <vector>

using std::vector;

using InitializeFunction = void (*)(ProfileParams &profile_params,
                                    vector<double> &state);
using InitializeSensitivityFunction =
    void (*)(ProfileParams &profile_params, vector<double> &state_sensitivity);
using RhsFunction = double (*)(const vector<double> &state, int state_offset,
                               const vector<double> &field, int field_offset,
                               vector<double> &rhs,
                               const ProfileParams &profile_params);
using RhsJacobianFunction = void (*)(const vector<double> &state,
                                     int state_offset,
                                     const vector<double> &field,
                                     int field_offset,
                                     vector<double> &matrix_data,
                                     const ProfileParams &profile_params);
using LimitUpdateFunction = double (*)(const vector<double> &state,
                                       const vector<double> &state_varn,
                                       const ProfileParams &profile_params);
using ComputeOutputsFunction = void (*)(const vector<double> &state_grid,
                                        const vector<double> &eta_grid,
                                        vector<double> &output_grid,
                                        size_t profile_size,
                                        const ProfileParams &profile_params);

class BLModel {
public:
  BLModel(InitializeFunction init_fun,
          InitializeSensitivityFunction init_sensitivity_fun,
          RhsFunction rhs_self_similar_fun, RhsFunction rhs_locally_similar_fun,
          RhsFunction rhs_diff_diff_fun,
          RhsJacobianFunction jacobian_self_similar_fun,
          RhsJacobianFunction jacobian_locally_similar_fun,
          RhsJacobianFunction jacobian_diff_diff_fun,
          LimitUpdateFunction limit_update_fun,
          ComputeOutputsFunction compute_outputs_fun)
      : initialize(init_fun), initialize_sensitivity(init_sensitivity_fun),
        compute_rhs_self_similar(rhs_self_similar_fun),
        compute_rhs_locally_similar(rhs_locally_similar_fun),
        compute_rhs_diff_diff(rhs_diff_diff_fun),
        compute_rhs_jacobian_self_similar(jacobian_self_similar_fun),
        compute_rhs_jacobian_locally_similar(jacobian_locally_similar_fun),
        compute_rhs_jacobian_diff_diff(jacobian_diff_diff_fun),
        limit_update(limit_update_fun), compute_outputs(compute_outputs_fun){};

  InitializeFunction initialize;
  InitializeSensitivityFunction initialize_sensitivity;
  LimitUpdateFunction limit_update;
  RhsFunction compute_rhs_self_similar, compute_rhs_locally_similar,
      compute_rhs_diff_diff;
  RhsJacobianFunction compute_rhs_jacobian_self_similar,
      compute_rhs_jacobian_locally_similar, compute_rhs_jacobian_diff_diff;
  ComputeOutputsFunction compute_outputs;
};

#endif