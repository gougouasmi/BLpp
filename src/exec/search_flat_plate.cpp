#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "flat_plate.h"
#include "profile_search.h"

int main(int argc, char *argv[]) {

  SearchWindow window;
  window.SetDefault();
  window.ParseCmdInputs(argc, argv);

  SearchParams params;
  params.SetDefault();
  params.ParseCmdInputs(argc, argv);

  std::vector<double> best_guess(2, 0.0);

  FlatPlate flat_plate(2000);

  flat_plate.BoxProfileSearch(window, params, best_guess);

  return 0;
}