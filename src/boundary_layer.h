#ifndef BOUNDARY_LAYER_H
#define BOUNDARY_LAYER_H

#include "boundary_data_struct.h"
#include "profile_struct.h"
#include "search_struct.h"

#include <vector>

using std::vector;

class BoundaryLayer {
public:
  BoundaryLayer(int max_nb_steps);
  BoundaryLayer(int max_nb_steps, InitializeFunction init_fun,
                InitializeSensitivityFunction init_sensitivity_fun,
                RhsFunction rhs_self_similar_fun,
                RhsFunction rhs_locally_similar_fun,
                RhsFunction rhs_diff_diff_fun,
                RhsJacobianFunction jacobian_self_similar_fun,
                RhsJacobianFunction jacobian_locally_similar_fun,
                RhsJacobianFunction jacobian_diff_diff_fun);

  //
  void InitializeState(ProfileParams &profile_params, int worker_id = 0);

  //
  RhsFunction GetRhsFun(SolveType solve_type);
  RhsJacobianFunction GetJacobianFun(SolveType solve_type);

  // ODE integration (eta)
  int DevelopProfile(ProfileParams &profile_params, vector<double> &score,
                     int worker_id = 0);
  int DevelopProfileExplicit(ProfileParams &profile_params,
                             vector<double> &score, int worker_id = 0);
  int DevelopProfileImplicit(ProfileParams &profile_params,
                             vector<double> &score, int worker_id = 0);

  // Shooting algorithm implementations
  int ProfileSearch(ProfileParams &profile_params, SearchParams &search_params,
                    vector<double> &best_guess);
  int BoxProfileSearch(ProfileParams &profile_params,
                       SearchParams &search_params, vector<double> &best_guess);
  int BoxProfileSearchParallel(ProfileParams &profile_params,

                               SearchParams &search_params,
                               vector<double> &best_guess);
  int BoxProfileSearchParallelWithQueues(ProfileParams &profile_params,
                                         SearchParams &search_params,
                                         vector<double> &best_guess);
  int GradientProfileSearch(ProfileParams &profile_params,
                            SearchParams &search_params,
                            vector<double> &best_guess);

  // 2D profile calculation
  void Compute(const BoundaryData &boundary_data, ProfileParams &profile_params,
               SearchParams &search_params,
               vector<vector<double>> &bl_state_grid);
  void ComputeLS(const BoundaryData &boundary_data,
                 ProfileParams &profile_params, SearchParams &search_params,
                 vector<vector<double>> &bl_state_grid);
  void ComputeDD(const BoundaryData &boundary_data,
                 ProfileParams &profile_params, SearchParams &search_params,
                 vector<vector<double>> &bl_state_grid);

private:
  const int _max_nb_workers = 8;
  int _max_nb_steps;

  vector<vector<double>> state_grids;
  vector<vector<double>> eta_grids;
  vector<vector<double>> rhs_vecs;

  vector<vector<double>> sensitivity_matrices;
  vector<vector<double>> matrix_buffers;

  vector<double> field_grid;

  InitializeFunction initialize;
  InitializeSensitivityFunction initialize_sensitivity;
  RhsFunction compute_rhs_self_similar, compute_rhs_locally_similar,
      compute_rhs_diff_diff;
  RhsJacobianFunction compute_rhs_jacobian_self_similar,
      compute_rhs_jacobian_locally_similar, compute_rhs_jacobian_diff_diff;
};

#endif