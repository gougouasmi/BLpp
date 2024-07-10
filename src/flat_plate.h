#ifndef FLAT_PLATE_H
#define FLAT_PLATE_H

#include "profile.h"
#include "profile_search.h"

#include <vector>

class FlatPlate {
public:
  FlatPlate(int max_nb_steps);
  FlatPlate(int max_nb_steps, RhsFunction rhs_fun, InitializeFunction init_fun);

  void InitializeState(ProfileParams &ProfileParams);
  int DevelopProfile(ProfileParams &profile_params, std::vector<double> &score,
                     bool &converged);
  void BoxProfileSearch(SearchWindow &window, SearchParams &params,
                        std::vector<double> &best_guess);

private:
  int _max_nb_steps;

  std::vector<double> state_grid;
  std::vector<double> eta_grid;
  std::vector<double> rhs;

  RhsFunction compute_rhs;
  InitializeFunction initialize;
};

#endif