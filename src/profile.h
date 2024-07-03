#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
#include <vector>

static int SYSTEM_RANK = 5;

static int FPP_ID = 0;
static int GP_ID = 1;
static int FP_ID = 2;
static int F_ID = 3;
static int G_ID = 4;

typedef struct ProfileParams {
  int nb_steps;
  double fpp0;
  double gp0;

  double min_eta_step;

  bool valid() const {
    return (nb_steps > 1) && (fpp0 >= 0.) && (gp0 >= 0.) && (min_eta_step > 0);
  }

  void set_default() {
    nb_steps = 2000;
    fpp0 = 0.5;
    gp0 = 0.5;
    min_eta_step = 1e-2;
  }

  void set_initial_values(std::vector<double> &initial_vals) {
    fpp0 = initial_vals[0];
    gp0 = initial_vals[1];
  }

  void parse_cmd_inputs(int argc, char *argv[]) {
    for (int i = 1; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "-n") {
        if (i + 1 < argc) {
          nb_steps = std::stoi(argv[++i]);
        } else {
          printf("profile nb_steps spec is incomplete.\n");
        }
      } else if (arg == "-fpp0") {
        if (i + 1 < argc) {
          fpp0 = std::stod(argv[++i]);
        } else {
          printf("profile f''(0) spec is incomplete.\n");
        }
      } else if (arg == "-gp0") {
        if (i + 1 < argc) {
          gp0 = std::stod(argv[++i]);
        } else {
          printf("profile g'(0) spec is incomplete.\n");
        }
      } else if (arg == "-eta") {
        if (i + 1 < argc) {
          min_eta_step = std::stod(argv[++i]);
        } else {
          printf("profile eta spec is incomplete.\n");
        }
      }
    }
  }

} ProfileParams;

int develop_profile(ProfileParams &profile_params,
                    std::vector<double> &state_grid,
                    std::vector<double> &eta_grid, std::vector<double> &rhs,
                    std::vector<double> &score, bool &converged);

#endif