#ifndef SEARCH_STRUCT_HPP
#define SEARCH_STRUCT_HPP

#include <array>
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "parsing.hpp"

using std::array;
using std::string;

enum class SearchMethod {
  BoxSerial,
  BoxParallel,
  BoxParallelQueue,
  GradientSerial,
  GradientExp,
};
const std::map<string, SearchMethod> SEARCH_KEYS = {
    {"box_serial", SearchMethod::BoxSerial},
    {"box_parallel", SearchMethod::BoxParallel},
    {"box_parallel_queue", SearchMethod::BoxParallelQueue},
    {"gradient", SearchMethod::GradientSerial},
    {"gradient_exp", SearchMethod::GradientExp},
};
static std::optional<SearchMethod> search_from_string(const string &key) {
  if (SEARCH_KEYS.count(key)) {
    SearchMethod search = SEARCH_KEYS.at(key);
    return search;
  }
  return {};
}
static inline string to_string(const SearchMethod &method) {
  switch (method) {
  case SearchMethod::BoxSerial:
    return "Box serial";
  case SearchMethod::BoxParallel:
    return "Box parallel";
  case SearchMethod::BoxParallelQueue:
    return "Box parallel w queues";
  case SearchMethod::GradientSerial:
    return "Gradient serial";
  case SearchMethod::GradientExp:
    return "Gradient experimental";
  default:
    return "method not recognized";
  }
}

//
enum class Scoring { Default, Square, SquareSteady, Exp, ExpScaled };
const std::map<string, Scoring> SCORING_KEYS = {
    {"square", Scoring::Square},
    {"square_steady", Scoring::SquareSteady},
    {"exp", Scoring::Exp},
    {"exp_scaled", Scoring::ExpScaled},
};
static std::optional<Scoring> scoring_from_string(const string &key) {
  if (SCORING_KEYS.count(key)) {
    Scoring scoring = SCORING_KEYS.at(key);
    return scoring;
  }
  return {};
}

struct SearchOutcome {
  bool success;
  int worker_id;
  int profile_size;
  array<double, 2> guess{{0.0, 0.0}};
};

static std::ofstream &operator<<(std::ofstream &s,
                                 const SearchOutcome &outcome) {
  s << outcome.success << ", " << outcome.worker_id << ", "
    << outcome.profile_size << ", " << outcome.guess[0] << ", "
    << outcome.guess[1] << "\n";
  return s;
}

static void WriteOutcomes(const vector<SearchOutcome> &outcomes,
                          const char *filepath = "search_outcomes.csv") {
  int nb_outcomes = outcomes.size();
  assert(nb_outcomes > 0);

  std::ofstream file(filepath);
  assert(file.is_open());

  for (const auto &outcome : outcomes) {
    file << outcome;
  }

  file.close();
}

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
    ParseValues(argc, argv, {{"-xdim", &xdim}, {"-ydim", &ydim}});
    ParseValues(argc, argv,
                {{"-x0", &fpp_min},
                 {"-x1", &fpp_max},
                 {"-y0", &gp_min},
                 {"-y1", &gp_max}});
    assert(xdim > 1);
    assert(ydim > 1);
    assert(fpp_max > fpp_min);
    assert(gp_max > gp_min);
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

  /*
   Read search parameters from argument list.

   -maxiter <number_of_iterations> -rtol <residual_tolerance> -v (verbosity)
   -search_method -scoring_method

   @param argc number of arguments,
   @param argv list of command line arguments,
  */
  void ParseCmdInputs(int argc, char *argv[]) {
    ParseValues<SearchMethod>(argc, argv, {{"-search", &method}},
                              search_from_string);
    ParseValues<Scoring>(argc, argv, {{"-scoring", &scoring}},
                         scoring_from_string);
    ParseValues(argc, argv, {{"-maxiter", &max_iter}});
    ParseValues(argc, argv, {{"-rtol", &rtol}});
    ParseOptions(argc, argv, {{"-v", &verbose}});

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