#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "profile_search.h"

int main(int argc, char *argv[]) {

  SearchWindow window;
  window.set_default();
  window.parse_cmd_inputs(argc, argv);
  window.print();

  SearchParams params;
  params.set_default();
  params.parse_cmd_inputs(argc, argv);

  std::vector<double> best_guess(2, 0.0);

  box_profile_search(window, params, best_guess);

  return 0;
}