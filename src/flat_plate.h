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

  void InitializeState(ProfileParams &profile_params, int worker_id = 0);

  bool DevelopProfile(ProfileParams &profile_params, vector<double> &score,
                      int worker_id = 0);
  bool DevelopProfileExplicit(ProfileParams &profile_params,
                              vector<double> &score, int worker_id = 0);
  bool DevelopProfileImplicit(ProfileParams &profile_params,
                              vector<double> &score, int worker_id = 0);

  void BoxProfileSearch(ProfileParams &profile_params, SearchWindow &window,
                        SearchParams &params, vector<double> &best_guess);
  void BoxProfileSearchParallel(ProfileParams &profile_params,
                                SearchWindow &window, SearchParams &params,
                                vector<double> &best_guess);
  void BoxProfileSearchParallelWithQueues(ProfileParams &profile_params,
                                          SearchWindow &window,
                                          SearchParams &params,
                                          vector<double> &best_guess);

private:
  const int _max_nb_workers = 8;
  int _max_nb_steps;

  vector<vector<double>> state_grids;
  vector<vector<double>> eta_grids;
  vector<vector<double>> rhs_vecs;

  RhsFunction compute_rhs;
  RhsJacobianFunction compute_rhs_jacobian;
  InitializeFunction initialize;
};

#endif