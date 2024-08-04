#ifndef FLAT_PLATE_H
#define FLAT_PLATE_H

#include "profile.h"
#include "profile_search.h"

#include <vector>

using std::vector;

class FlatPlate {
public:
  FlatPlate(int max_nb_steps);
  FlatPlate(int max_nb_steps, RhsFunction rhs_fun, InitializeFunction init_fun,
            RhsJacobianFunction jacobian_fun);

  //
  void InitializeState(ProfileParams &profile_params, int worker_id = 0);

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
  void ComputeLS(const vector<double> &edge_field, SearchParams &search_params,
                 vector<vector<double>> &bl_state_grid);
  void ComputeDD(const vector<double> &edge_field, SearchParams &search_params,
                 vector<vector<double>> &bl_state_grid);

private:
  const int _max_nb_workers = 8;
  int _max_nb_steps;

  vector<vector<double>> state_grids;
  vector<vector<double>> eta_grids;
  vector<vector<double>> rhs_vecs;

  vector<double> field_grid;

  RhsFunction compute_rhs;
  RhsJacobianFunction compute_rhs_jacobian;
  InitializeFunction initialize;
};

#endif