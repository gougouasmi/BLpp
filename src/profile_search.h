#ifndef PROFILE_SEARCH_H
#define PROFILE_SEARCH_H

#include <cassert>
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
          assert(fpp_min < fpp_max);
        } else {
          printf("fbounds spec is incomplete. Setting default.");
        }
      } else if (arg == "-gbounds") {
        if (i + 2 < argc) {
          gp_min = std::stod(argv[++i]);
          gp_max = std::stod(argv[++i]);
          assert(gp_min < gp_max);
        } else {
          printf("gbounds spec is incomplete. Setting default.");
        }
      } else if (arg == "-boxdims") {
        if (i + 2 < argc) {
          xdim = std::stoi(argv[++i]);
          ydim = std::stoi(argv[++i]);
          assert(xdim > 1);
          assert(ydim > 1);
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
          assert(max_iter >= 1);
        }
      } else if (arg == "-rtol") {
        if (i + 1 < argc) {
          rtol = std::stod(argv[++i]);
          assert(rtol > 0);
        }
      } else if (arg == "-v") {
        verbose = true;
      }
    }
  }

} SearchParams;

struct SearchInput {
  double x0;
  double dx;
  double y0;
  double dy;

  int xid_start;
  int xid_end;
  int yid_start;
  int yid_end;

  SearchInput(double x0_val, double dx_val, double y0_val, double dy_val,
              int xid_start_val, int xid_end_val, int yid_start_val,
              int yid_end_val)
      : x0(x0_val), dx(dx_val), y0(y0_val), dy(dy_val),
        xid_start(xid_start_val), xid_end(xid_end_val),
        yid_start(yid_start_val), yid_end(yid_end_val){};
  static SearchInput StopMessage() {
    return SearchInput(0., 0., 0., 0., 0, 0, 0, 0);
  };
  bool Stop() { return (dx == 0.) || (dy == 0.); };
};

struct SearchResult {
  double res_norm;
  int xid;
  int yid;
  SearchResult(double res_val, int xid_val, int yid_val)
      : res_norm(res_val), xid(xid_val), yid(yid_val){};
  static SearchResult StopMessage() { return SearchResult(1e30, -1, -1); };
};

#endif
