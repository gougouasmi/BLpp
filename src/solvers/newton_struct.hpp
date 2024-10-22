#ifndef NEWTON_STRUCT_HPP
#define NEWTON_STRUCT_HPP

struct NewtonParams {
  double rtol = 1e-6;
  int max_iter = 1000;
  int max_ls_iter = 10;
  bool verbose = false;
};

#endif