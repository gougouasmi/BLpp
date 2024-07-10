#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
#include <vector>

const int FLAT_PLATE_RANK = 5;

const int FPP_ID = 0;
const int GP_ID = 1;
const int FP_ID = 2;
const int F_ID = 3;
const int G_ID = 4;

enum WallType { Wall, Adiabatic };

typedef struct ProfileParams {
  int nb_steps;

  WallType wall_type;
  double fpp0;
  double gp0;
  double g0;

  double min_eta_step;

  double pe = 1.;
  double he = 1.;
  double roe = 1.;
  double ue = 1.;
  double mue = 1.;

  double eckert = 1.;

  bool AreValid() const {
    return (nb_steps >= 1) && (fpp0 >= 0.) && (gp0 >= 0.) && (min_eta_step > 0);
  }

  void SetDefault() {
    nb_steps = 2000;
    fpp0 = 0.5;
    gp0 = 0.5;
    wall_type = WallType::Wall;
    g0 = 0.2;
    min_eta_step = 1e-2;
  }

  void SetInitialValues(std::vector<double> &initial_vals) {
    fpp0 = initial_vals[0];

    switch (wall_type) {
    case WallType::Wall:
      gp0 = initial_vals[1];
      break;
    case WallType::Adiabatic:
      g0 = initial_vals[1];
      break;
    default:
      printf("\nwall_type not recognized\n");
      break;
    }
  }

  void ParseCmdInputs(int argc, char *argv[]) {
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
      } else if (arg == "-g0") {
        if (i + 1 < argc) {
          g0 = std::stod(argv[++i]);
        } else {
          printf("profile g(0) spec is incomplete.\n");
        }
      } else if (arg == "-eta") {
        if (i + 1 < argc) {
          min_eta_step = std::stod(argv[++i]);
        } else {
          printf("profile eta spec is incomplete.\n");
        }
      } else if (arg == "-wadiab") {
        wall_type = WallType::Adiabatic;
      }
    }
  }
} ProfileParams;

double compute_rhs_default(std::vector<double> &state, std::vector<double> &rhs,
                           int offset, ProfileParams &params);
double compute_rhs_cpg(std::vector<double> &state, std::vector<double> &rhs,
                       int offset, ProfileParams &params);

void initialize_default(ProfileParams &profile_params,
                        std::vector<double> &state);
void initialize_cpg(ProfileParams &profile_params, std::vector<double> &state);

typedef double (*RhsFunction)(std::vector<double> &state,
                              std::vector<double> &rhs, int offset,
                              ProfileParams &profile_params);
typedef void (*InitializeFunction)(ProfileParams &profile_params,
                                   std::vector<double> &state);

#endif