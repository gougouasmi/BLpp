#ifndef PROFILE_SEARCH_H
#define PROFILE_SEARCH_H

#include <iostream>
#include <vector>

typedef struct SearchWindow {
  double fpp_min;
  double fpp_max;
  double gp_min;
  double gp_max;
  int xdim;
  int ydim;

  void SetDefault() {
    fpp_min = 0.01;
    fpp_max = 10.0;
    gp_min = 0.1;
    gp_max = 10.0;
    xdim = 10;
    ydim = 10;
  }

  void Print() const {
    printf("Window = {fpp_min=%.2e, fpp_max=%.2e, gp_min=%.2e, gp_max=%.2e, "
           "xdim=%d, ydim=%d}.\n\n",
           fpp_min, fpp_max, gp_min, gp_max, xdim, ydim);
  }

  void ParseCmdInputs(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-fbounds") {
        if (i + 2 < argc) {
          fpp_min = std::stod(argv[++i]);
          fpp_max = std::stod(argv[++i]);
        } else {
          printf("fbounds spec is incomplete. Setting default.");
        }
      } else if (arg == "-gbounds") {
        if (i + 2 < argc) {
          gp_min = std::stod(argv[++i]);
          gp_max = std::stod(argv[++i]);
        } else {
          printf("gbounds spec is incomplete. Setting default.");
        }
      } else if (arg == "-boxdim") {
        if (i + 2 < argc) {
          xdim = std::stoi(argv[++i]);
          ydim = std::stoi(argv[++i]);
        }
      }
    }
  }

} SearchWindow;

typedef struct SearchParams {
  int max_iter;
  bool verbose;
  double rtol;

  void SetDefault() {
    max_iter = 20;
    rtol = 1e-3;
    verbose = false;
  }

  void ParseCmdInputs(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-maxiter") {
        if (i + 1 < argc) {
          max_iter = std::stoi(argv[++i]);
        }
      } else if (arg == "-rtol") {
        if (i + 1 < argc) {
          rtol = std::stod(argv[++i]);
        }
      } else if (arg == "-v") {
        verbose = true;
      }
    }
  }

} SearchParams;

#endif
