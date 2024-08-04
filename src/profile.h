#ifndef PROFILE_H
#define PROFILE_H

#include <iostream>
#include <vector>

constexpr int FLAT_PLATE_RANK = 5;

constexpr int FPP_ID = 0;
constexpr int GP_ID = 1;
constexpr int FP_ID = 2;
constexpr int F_ID = 3;
constexpr int G_ID = 4;

constexpr int FIELD_RANK = 8;

enum WallType { Wall, Adiabatic };
enum TimeScheme { Explicit, Implicit };

typedef struct ProfileParams {
  int nb_steps;

  WallType wall_type;
  double fpp0;
  double gp0;
  double g0;

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

  bool AreValid() const {
    return (nb_steps >= 1) && (fpp0 >= 0.) && (gp0 >= 0.) && (max_step > 0);
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
      }
    }
  }
} ProfileParams;

double compute_rhs_default(const std::vector<double> &state, int state_offset,
                           const std::vector<double> &field, int field_offset,
                           std::vector<double> &rhs, ProfileParams &params);
double compute_rhs_cpg(const std::vector<double> &state, int state_offset,
                       const std::vector<double> &field, int field_offset,
                       std::vector<double> &rhs, ProfileParams &params);

void compute_rhs_jacobian_default(const std::vector<double> &state,
                                  const std::vector<double> &field,
                                  int field_offset,
                                  std::vector<double> &matrix_data,
                                  ProfileParams &params);
void compute_rhs_jacobian_cpg(const std::vector<double> &state,
                              const std::vector<double> &field,
                              int field_offset,
                              std::vector<double> &matrix_data,
                              ProfileParams &params);

void initialize_default(ProfileParams &profile_params,
                        std::vector<double> &state);
void initialize_cpg(ProfileParams &profile_params, std::vector<double> &state);

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