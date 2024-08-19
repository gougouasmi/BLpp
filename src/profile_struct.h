#ifndef PROFILE_STRUCT_H
#define PROFILE_STRUCT_H

#include <iostream>
#include <vector>

constexpr int BL_RANK = 5;

constexpr int FPP_ID = 0;
constexpr int GP_ID = 1;
constexpr int FP_ID = 2;
constexpr int F_ID = 3;
constexpr int G_ID = 4;

constexpr int FIELD_RANK = 8;

enum WallType { Wall, Adiabatic };
enum TimeScheme { Explicit, Implicit };
enum SolveType { SelfSimilar, LocallySimilar, DifferenceDifferential };

typedef struct ProfileParams {
  int nb_steps;

  WallType wall_type;
  double fpp0;
  double gp0;
  double g0;

  SolveType solve_type = SolveType::SelfSimilar;

  TimeScheme scheme = TimeScheme::Explicit;
  double max_step;

  double ue = 1.;
  double he = 1.;
  double pe = 1.;

  double xi = 0.;
  double due_dxi = 0.;
  double dhe_dxi = 0.;

  double roe = 1.;
  double mue = 1.;

  double eckert = 1.;

  void PrintEdgeValues() const {
    printf("Profile parameters: -ue=%.2e, -he=%.2e, -pe=%.2e, -xi=%.2e, "
           "-due_dxi=%.2e, -dhe_dxi=%.2e, -eckert=%.2e.\n",
           ue, he, pe, xi, due_dxi, dhe_dxi, eckert);
  }

  void PrintODEFactors() const {
    printf("Profile ODE factors: eckert = %.2e, (xi / ue) * due_dxi = %.2e, xi "
           "* dhe_dxi / he "
           "= %.2e, xi * (ue / he) * due_dxi = %.2e.\n",
           eckert, (xi / ue) * due_dxi, xi * dhe_dxi / he,
           xi * (ue / he) * due_dxi);
  }

  bool AreValid() const {
    if ((wall_type == WallType::Adiabatic) && (gp0 != 0.)) {
      printf("Adiabatic wall yet g'(0) != 0.\n");
      return false;
    }

    if (solve_type == SolveType::LocallySimilar ||
        solve_type == SolveType::DifferenceDifferential) {
      if (ue == 0) {
        printf("ue = 0 singularity.\n");
        return false;
      }
    }

    if (he <= 0) {
      printf("h_{e} cannot be <=0 (CPG).\n");
      return false;
    }

    if (g0 <= 0) {
      printf("g(0) = h/h_{e} cannot be <= 0.\n");
      return false;
    }

    return (nb_steps > 1) && (fpp0 >= 0.) && (max_step > 0);
  }

  void SetDefault() {
    nb_steps = 2000;
    fpp0 = 0.5;
    gp0 = 0.5;
    wall_type = WallType::Wall;
    g0 = 0.2;
    max_step = 1e-2;
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
          max_step = std::stod(argv[++i]);
        } else {
          printf("profile eta spec is incomplete.\n");
        }
      } else if (arg == "-wadiab") {
        wall_type = WallType::Adiabatic;
      } else if (arg == "-implicit") {
        scheme = TimeScheme::Implicit;
      } else if (arg == "-local_sim") {
        solve_type = SolveType::LocallySimilar;
      } else if (arg == "-diff_diff") {
        solve_type = SolveType::DifferenceDifferential;
      }
    }
  }
} ProfileParams;

typedef void (*InitializeFunction)(ProfileParams &profile_params,
                                   std::vector<double> &state);

typedef double (*RhsFunction)(const std::vector<double> &state,
                              int state_offset,
                              const std::vector<double> &field,
                              int field_offset, std::vector<double> &rhs,
                              ProfileParams &profile_params);
typedef void (*RhsJacobianFunction)(const std::vector<double> &state,
                                    const std::vector<double> &field,
                                    int field_offset,
                                    std::vector<double> &matrix_data,
                                    ProfileParams &profile_params);

#endif