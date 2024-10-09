#ifndef PROFILE_STRUCT_H
#define PROFILE_STRUCT_H

#include <array>
#include <iostream>
#include <map>
#include <vector>

#include "boundary_data_struct.h"
#include "indexing.h"
#include "parsing.h"
#include "search_struct.h"

using std::array;
using std::vector;

// State variable indices
constexpr int BL_RANK = 5;

constexpr int FPP_ID = 0;
constexpr int GP_ID = 1;
constexpr int FP_ID = 2;
constexpr int F_ID = 3;
constexpr int G_ID = 4;

constexpr std::array<int, BL_RANK> BL_INDICES{FPP_ID, GP_ID, FP_ID, F_ID, G_ID};
static_assert(complete_indexing(BL_INDICES));

// Field variable indices (Difference-Differential method)
constexpr int FIELD_RANK = 6;

constexpr int FIELD_M0_ID = 0;
constexpr int FIELD_M1_ID = 1;
constexpr int FIELD_S0_ID = 2;
constexpr int FIELD_S1_ID = 3;
constexpr int FIELD_E0_ID = 4;
constexpr int FIELD_E1_ID = 5;

constexpr std::array<int, FIELD_RANK> FIELD_INDICES{FIELD_M0_ID, FIELD_M1_ID,
                                                    FIELD_S0_ID, FIELD_S1_ID,
                                                    FIELD_E0_ID, FIELD_E1_ID};
static_assert(complete_indexing(FIELD_INDICES));

// Output field indices
constexpr int OUTPUT_RANK = 7;

constexpr int OUTPUT_TAU_ID = 0;
constexpr int OUTPUT_Q_ID = 1;
constexpr int OUTPUT_RO_ID = 2;
constexpr int OUTPUT_Y_ID = 3;

constexpr int OUTPUT_MU_ID = 4;
constexpr int OUTPUT_PRANDTL_ID = 5;
constexpr int OUTPUT_CHAPMANN_ID = 6;

constexpr std::array<int, OUTPUT_RANK> OUTPUT_INDICES{
    OUTPUT_TAU_ID, OUTPUT_Q_ID,       OUTPUT_RO_ID,      OUTPUT_Y_ID,
    OUTPUT_MU_ID,  OUTPUT_PRANDTL_ID, OUTPUT_CHAPMANN_ID};
static_assert(complete_indexing(OUTPUT_INDICES));

//
enum class WallType { Wall, Adiabatic };
const std::map<string, WallType> WALL_STRINGS = {
    {"adiab", WallType::Adiabatic},
};
static std::optional<WallType> wall_type_from_string(const string &key) {
  if (WALL_STRINGS.count(key)) {
    WallType wall_type = WALL_STRINGS.at(key);
    return wall_type;
  }
  return {};
}

//
enum class TimeScheme { Explicit, Implicit, ImplicitCrankNicolson };
constexpr array<double, 3> SCHEME_THETA = {{0., 1., 0.5}};
const std::map<string, TimeScheme> SCHEME_STRINGS = {
    {"explicit", TimeScheme::Explicit},
    {"implicit", TimeScheme::Implicit},
    {"implicit_cn", TimeScheme::ImplicitCrankNicolson}};
static std::optional<TimeScheme> scheme_from_string(const string &key) {
  if (SCHEME_STRINGS.count(key)) {
    TimeScheme scheme = SCHEME_STRINGS.at(key);
    return scheme;
  }
  return {};
}

//
enum class SolveType { SelfSimilar, LocallySimilar, DifferenceDifferential };
const std::map<string, SolveType> SOLVE_TYPE_STRINGS = {
    {"self_sim", SolveType::SelfSimilar},
    {"loc_sim", SolveType::LocallySimilar},
    {"diff_diff", SolveType::DifferenceDifferential}};
static std::optional<SolveType> solve_type_from_string(const string &key) {
  if (SOLVE_TYPE_STRINGS.count(key)) {
    SolveType solve_type = SOLVE_TYPE_STRINGS.at(key);
    return solve_type;
  }
  return {};
}

//
struct ProfileParams {
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
  double ue{1.};
  double he{1.};
  double pe{1.};

  double xi{0.};
  double due_dxi{0.};
  double dhe_dxi{0.};

  // Secondary edge conditions (which
  // you can compute from primary)
  double roe{1.};
  double mue{1.};

  double eckert{1.};

  double c1{0.};
  double c2{0.};
  double c3{0.};

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

    c1 = 2. * (xi / ue) * due_dxi;
    c2 = 2. * xi * dhe_dxi / he;
    c3 = 2. * xi * (ue / he) * due_dxi;
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
             eckert, c1, c2, c3);
    }

    if (solve_type == SolveType::DifferenceDifferential) {
      printf("Profile ODE factors (difference-differential):\n - eckert=%.2e, "
             "c1=%.2e, "
             "c2=%.2e, c3=%.2e\n\n",
             eckert, c1, c2, c3);
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

  void SetInitialValues(array<double, 2> &initial_vals) {
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

  /*
   Read profile parameters from argument list.

   -n <eta_grid_size> -fpp0 <f''(0)> -gp0 <g'(0)> -g0 <g(0)>

   -eta <eta_step> -walltype -time_scheme -solve_type

   @param argc number of arguments,
   @param argv list of arguments,
  */
  void ParseCmdInputs(int argc, char *argv[]) {
    ParseValues(argc, argv, {{"-n", &nb_steps}});
    ParseValues(
        argc, argv,
        {{"-fpp0", &fpp0}, {"-gp0", &gp0}, {"-g0", &g0}, {"-eta", &max_step}});
    ParseValues<TimeScheme>(argc, argv, {{"-eta_scheme", &scheme}},
                            scheme_from_string);
    ParseValues<SolveType>(argc, argv, {{"-solve_type", &solve_type}},
                           solve_type_from_string);
    ParseValues<WallType>(argc, argv, {{"-wall", &wall_type}},
                          wall_type_from_string);
  }
};

#endif