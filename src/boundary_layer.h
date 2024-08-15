#ifndef BOUNDARY_LAYER_H
#define BOUNDARY_LAYER_H

#include "profile_struct.h"
#include "search_struct.h"

#include <vector>

using std::vector;

class BoundaryLayer {
public:
  BoundaryLayer(int max_nb_steps);
  BoundaryLayer(int max_nb_steps, InitializeFunction init_fun,
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
  bool DevelopProfile(ProfileParams &profile_params, vector<double> &score,
                      int worker_id = 0);
  bool DevelopProfileExplicit(ProfileParams &profile_params,
                              vector<double> &score, int worker_id = 0);
  bool DevelopProfileImplicit(ProfileParams &profile_params,
                              vector<double> &score, int worker_id = 0);

  // Shooting algorithm implementations
  int BoxProfileSearch(ProfileParams &profile_params, SearchWindow &window,
                       SearchParams &params, vector<double> &best_guess);
  int BoxProfileSearchParallel(ProfileParams &profile_params,
                               SearchWindow &window, SearchParams &params,
                               vector<double> &best_guess);
  int BoxProfileSearchParallelWithQueues(ProfileParams &profile_params,
                                         SearchWindow &window,
                                         SearchParams &params,
                                         vector<double> &best_guess);

  // 2D profile calculation
  void Compute(const vector<double> &edge_field,
               const vector<double> &wall_field, ProfileParams &profile_params,
               SearchParams &search_params,
               vector<vector<double>> &bl_state_grid);
  void ComputeLS(const vector<double> &edge_field,
                 const vector<double> &wall_field,
                 ProfileParams &profile_params, SearchParams &search_params,
                 vector<vector<double>> &bl_state_grid);
  void ComputeDD(const vector<double> &edge_field,
                 const vector<double> &wall_field,
                 ProfileParams &profile_params, SearchParams &search_params,
                 vector<vector<double>> &bl_state_grid);

private:
  const int _max_nb_workers = 8;
  int _max_nb_steps;

  vector<vector<double>> state_grids;
  vector<vector<double>> eta_grids;
  vector<vector<double>> rhs_vecs;

  vector<double> field_grid;

  InitializeFunction initialize;
  RhsFunction compute_rhs_self_similar, compute_rhs_locally_similar,
      compute_rhs_diff_diff;
  RhsJacobianFunction compute_rhs_jacobian_self_similar,
      compute_rhs_jacobian_locally_similar, compute_rhs_jacobian_diff_diff;
};

#endif