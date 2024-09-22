#ifndef SEARCH_STRUCT_H
#define SEARCH_STRUCT_H

#include <cassert>
#include <iostream>
#include <vector>

enum class SearchMethod {
  BoxSerial,
  BoxParallel,
  BoxParallelQueue,
  GradientSerial
};
enum class Scoring { Default, Square, SquareSteady, Exp, ExpScaled };

struct SearchOutcome {
  bool success;
  int worker_id;
  int profile_size;
};

struct SearchWindow {
  double fpp_min{0.1};
  double fpp_max{10.0};
  double gp_min{0.1};
  double gp_max{10.0};
  int xdim{10};
  int ydim{10};

  SearchWindow() = default;

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
};

struct SearchParams {
  int max_iter{20};
  bool verbose{false};
  double rtol{1e-3};

  SearchWindow window{};
  SearchMethod method{SearchMethod::GradientSerial};

  Scoring scoring{Scoring::Default};

  SearchParams() = default;

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
      } else if (arg == "-box_search") {
        method = SearchMethod::BoxSerial;
      } else if (arg == "-box_search_parallel") {
        method = SearchMethod::BoxParallel;
      } else if (arg == "-box_search_parallel_queue") {
        method = SearchMethod::BoxParallelQueue;
      } else if (arg == "-gradient_search") {
        method = SearchMethod::GradientSerial;
      } else if (arg == "-score_square") {
        scoring = Scoring::Square;
      } else if (arg == "-score_square_steady") {
        scoring = Scoring::SquareSteady;
      } else if (arg == "-score_exp") {
        scoring = Scoring::Exp;
      } else if (arg == "-score_exp_scaled") {
        scoring = Scoring::ExpScaled;
      }
    }
    window.ParseCmdInputs(argc, argv);
  }
};

struct BoxSearchInput {
  double x0;
  double dx;
  double y0;
  double dy;

  int xid_start;
  int xid_end;
  int yid_start;
  int yid_end;

  BoxSearchInput(double x0_val, double dx_val, double y0_val, double dy_val,
                 int xid_start_val, int xid_end_val, int yid_start_val,
                 int yid_end_val)
      : x0(x0_val), dx(dx_val), y0(y0_val), dy(dy_val),
        xid_start(xid_start_val), xid_end(xid_end_val),
        yid_start(yid_start_val), yid_end(yid_end_val){};
  static BoxSearchInput StopMessage() {
    return BoxSearchInput(0., 0., 0., 0., 0, 0, 0, 0);
  };
  bool Stop() { return (dx == 0.) || (dy == 0.); };
};

struct BoxSearchResult {
  double res_norm;
  int xid;
  int yid;
  int worker_id;
  int profile_size;
  BoxSearchResult(double res_val, int xid_val, int yid_val, int worker_id_val,
                  int profile_size_val)
      : res_norm(res_val), xid(xid_val), yid(yid_val), worker_id(worker_id_val),
        profile_size(profile_size_val){};
  static BoxSearchResult StopMessage() {
    return BoxSearchResult(1e30, -1, -1, -1, 1);
  };
};

#endif