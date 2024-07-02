#include <cstdlib>
#include <iostream>
#include <vector>

#include "profile_search.h"

int main(int argc, char *argv[]) {

  SearchWindow window;
  window.fpp_min = 0.2;
  window.fpp_max = 0.8;
  window.gp_min = 0.2;
  window.gp_max = 0.2;
  window.xdim = 10;
  window.ydim = 10;

  SearchParams params;
  params.max_iter = 10;
  params.rtol = 1e-2;
  params.verbose = true;

  std::vector<double> best_guess(2, 0.0);

  box_profile_search(window, params, best_guess);

  return 0;
}