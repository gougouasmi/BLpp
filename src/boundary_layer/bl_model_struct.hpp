#ifndef BL_MODEL_HPP
#define BL_MODEL_HPP

#include "generic_vector.hpp"
#include "profile_struct.hpp"

#include <vector>

using std::vector;

using InitializeFunction = void (*)(ProfileParams &profile_params,
                                    vector<double> &state);
using InitializeSensitivityFunction =
    void (*)(ProfileParams &profile_params, vector<double> &state_sensitivity);

template <std::size_t CTIME_RANK = 0>
using RhsFunction = double (*)(const Generic::Vector<double, CTIME_RANK> &state,
                               int state_offset, const vector<double> &field,
                               int field_offset,
                               Generic::Vector<double, CTIME_RANK> &rhs,
                               const ProfileParams &profile_params);

template <std::size_t CTIME_RANK = 0>
using RhsJacobianFunction =
    void (*)(const Generic::Vector<double, CTIME_RANK> &state, int state_offset,
             const vector<double> &field, int field_offset,
             Generic::Vector<double, CTIME_RANK> &matrix_data,
             const ProfileParams &profile_params);

template <std::size_t CTIME_RANK = 0>
using LimitUpdateFunction =
    double (*)(const Generic::Vector<double, CTIME_RANK> &state,
               const Generic::Vector<double, CTIME_RANK> &state_varn,
               const ProfileParams &profile_params);
using ComputeOutputsFunction = void (*)(const vector<double> &state_grid,
                                        const vector<double> &eta_grid,
                                        vector<double> &output_grid,
                                        size_t profile_size,
                                        const ProfileParams &profile_params);

template <std::size_t CTIME_RANK = 0> class BLModel {
public:
  BLModel(InitializeFunction init_fun,
          InitializeSensitivityFunction init_sensitivity_fun,
          RhsFunction<CTIME_RANK> rhs_self_similar_fun,
          RhsFunction<CTIME_RANK> rhs_locally_similar_fun,
          RhsFunction<CTIME_RANK> rhs_diff_diff_fun,
          RhsJacobianFunction<CTIME_RANK> jacobian_self_similar_fun,
          RhsJacobianFunction<CTIME_RANK> jacobian_locally_similar_fun,
          RhsJacobianFunction<CTIME_RANK> jacobian_diff_diff_fun,
          LimitUpdateFunction<CTIME_RANK> limit_update_fun,
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
  LimitUpdateFunction<CTIME_RANK> limit_update;
  RhsFunction<CTIME_RANK> compute_rhs_self_similar, compute_rhs_locally_similar,
      compute_rhs_diff_diff;
  RhsJacobianFunction<CTIME_RANK> compute_rhs_jacobian_self_similar,
      compute_rhs_jacobian_locally_similar, compute_rhs_jacobian_diff_diff;
  ComputeOutputsFunction compute_outputs;
};

#endif