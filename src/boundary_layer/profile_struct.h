#ifndef PROFILE_STRUCT_H
#define PROFILE_STRUCT_H

#include <iostream>
#include <vector>

#include "boundary_data_struct.h"
#include "search_struct.h"

using std::vector;

// State variable indices
constexpr int BL_RANK = 5;

constexpr int FPP_ID = 0;
constexpr int GP_ID = 1;
constexpr int FP_ID = 2;
constexpr int F_ID = 3;
constexpr int G_ID = 4;

// Field variable indices (Difference-Differential method)
constexpr int FIELD_RANK = 8;

// Output field indices
constexpr int OUTPUT_RANK = 7;

constexpr int OUTPUT_TAU_ID = 0;
constexpr int OUTPUT_Q_ID = 1;
constexpr int OUTPUT_RO_ID = 2;
constexpr int OUTPUT_Y_ID = 3;

constexpr int OUTPUT_MU_ID = 4;
constexpr int OUTPUT_PRANDTL_ID = 5;
constexpr int OUTPUT_CHAPMANN_ID = 6;

//
enum class WallType { Wall, Adiabatic };
enum class TimeScheme { Explicit, Implicit, ImplicitCrankNicolson };
enum class SolveType { SelfSimilar, LocallySimilar, DifferenceDifferential };

typedef struct ProfileParams {
  int nb_steps{2000};

  WallType wall_type{WallType::Wall};
  double fpp0{0.5};
  double gp0{0.5};
  double g0{0.2};

  SolveType solve_type{SolveType::SelfSimilar};

  TimeScheme scheme{TimeScheme::Explicit};
  double max_step{1e-2};

  Scoring scoring{Scoring::Default};

  // Primary edge conditions
  double ue = 1.;
  double he = 1.;
  double pe = 1.;

  double xi = 0.;
  double due_dxi = 0.;
  double dhe_dxi = 0.;

  // Secondary edge conditions (which
  // you can compute from primary)
  double roe = 1.;
  double mue = 1.;

  double eckert = 1.;

  //
  ProfileParams() = default;

  inline void ReadEdgeConditions(const vector<double> &edge_field,
                                 size_t offset) {
    ue = edge_field[offset + EDGE_U_ID];
    he = edge_field[offset + EDGE_H_ID];
    pe = edge_field[offset + EDGE_P_ID];
    xi = edge_field[offset + EDGE_XI_ID];
    due_dxi = edge_field[offset + EDGE_DU_DXI_ID];
    dhe_dxi = edge_field[offset + EDGE_DH_DXI_ID];
  }

  inline void ReadWallConditions(const vector<double> &wall_field,
                                 size_t offset) {
    g0 = wall_field[offset];
  }

  void PrintEdgeValues() const {
    printf("Profile parameters: -ue=%.2e, -he=%.2e, -pe=%.2e, -xi=%.2e, "
           "-due_dxi=%.2e, -dhe_dxi=%.2e, -eckert=%.2e.\n",
           ue, he, pe, xi, due_dxi, dhe_dxi, eckert);
  }

  void PrintODEFactors() const {

    if (solve_type == SolveType::SelfSimilar) {
      printf("Profile ODE factors (self-similar): eckert = %.2e.\n", eckert);
    }

    if (solve_type == SolveType::LocallySimilar) {
      printf("Profile ODE factors (locally-similar):\n - eckert=%.2e, c1=%.2e, "
             "c2=%.2e, c3=%.2e\n\n",
             eckert, 2. * (xi / ue) * due_dxi, 2. * xi * dhe_dxi / he,
             2. * xi * (ue / he) * due_dxi);
    }
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

    if ((fpp0 < 0.) || (isnan(fpp0))) {
      printf("f''(0) = %.2e (negative or nan).\n", fpp0);
      return false;
    }

    return (nb_steps > 1) && (max_step > 0);
  }

  void SetInitialValues(vector<double> &initial_vals) {
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
          printf("f''(0) set to %.2e.\n", fpp0);
        } else {
          printf("profile f''(0) spec is incomplete.\n");
        }
      } else if (arg == "-gp0") {
        if (i + 1 < argc) {
          gp0 = std::stod(argv[++i]);
          printf("g'(0) set to %.2e.\n", gp0);
        } else {
          printf("profile g'(0) spec is incomplete.\n");
        }
      } else if (arg == "-g0") {
        if (i + 1 < argc) {
          g0 = std::stod(argv[++i]);
          printf("g(0) set to %.2e.\n", g0);
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
      } else if (arg == "-explicit") {
        scheme = TimeScheme::Explicit;
      } else if (arg == "-implicit") {
        scheme = TimeScheme::Implicit;
      } else if (arg == "-implicit_cn") {
        scheme = TimeScheme::ImplicitCrankNicolson;
      } else if (arg == "-local_sim") {
        solve_type = SolveType::LocallySimilar;
      } else if (arg == "-diff_diff") {
        solve_type = SolveType::DifferenceDifferential;
      }
    }
  }
} ProfileParams;

#endif