#ifndef PROFILE_SEARCH_H
#define PROFILE_SEARCH_H

#include <vector>

typedef struct {
  double fpp_min;
  double fpp_max;
  double gp_min;
  double gp_max;
  int xdim;
  int ydim;
} SearchWindow;

typedef struct {
  int max_iter;
  bool verbose;
  double rtol;
} SearchParams;

void box_profile_search(SearchWindow &window, SearchParams &params,
                        std::vector<double> &best_guess);

#endif
