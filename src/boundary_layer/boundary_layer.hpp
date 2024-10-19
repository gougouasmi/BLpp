#ifndef BOUNDARY_LAYER_HPP
#define BOUNDARY_LAYER_HPP

#include "bl_model_struct.hpp"
#include "boundary_data_struct.hpp"
#include "newton_solver.hpp"
#include "profile_struct.hpp"
#include "search_struct.hpp"

#include <array>
#include <string>
#include <vector>

using std::array;
using std::vector;

class BoundaryLayer {
public:
  BoundaryLayer() = delete;
  BoundaryLayer(int max_nb_steps, BLModel model_functions);

  //
  void InitializeState(ProfileParams &profile_params, int worker_id = 0);

  //
  RhsFunction GetRhsFun(SolveType solve_type);
  RhsJacobianFunction GetJacobianFun(SolveType solve_type);

  // ODE integration (eta)
  int DevelopProfile(ProfileParams &profile_params, vector<double> &score,
                     int worker_id = 0, bool advise = false);
  int DevelopProfileExplicit(ProfileParams &profile_params, int worker_id = 0);
  int DevelopProfileImplicit(ProfileParams &profile_params, int worker_id = 0);
  int DevelopProfileImplicitCN(ProfileParams &profile_params,
                               int worker_id = 0);

  // Shooting algorithm implementations
  SearchOutcome ProfileSearch(ProfileParams &profile_params,
                              SearchParams &search_params,
                              array<double, 2> &best_guess);
  SearchOutcome BoxProfileSearch(ProfileParams &profile_params,
                                 SearchParams &search_params,
                                 array<double, 2> &best_guess);
  SearchOutcome BoxProfileSearchParallel(ProfileParams &profile_params,
                                         SearchParams &search_params,
                                         array<double, 2> &best_guess);
  SearchOutcome
  BoxProfileSearchParallelWithQueues(ProfileParams &profile_params,
                                     SearchParams &search_params,
                                     array<double, 2> &best_guess);

  // Gradient Descent / Newton method
  SearchOutcome GradientProfileSearch(ProfileParams &profile_params,
                                      SearchParams &search_params,
                                      array<double, 2> &best_guess,
                                      int worker_id = 0);

  SearchOutcome GradientProfileSearch_Exp(ProfileParams &profile_params,
                                          SearchParams &search_params,
                                          array<double, 2> &best_guess,
                                          int worker_id = 0);

  // 2D profile calculation
  vector<SearchOutcome> Compute(const BoundaryData &boundary_data,
                                ProfileParams &profile_params,
                                SearchParams &search_params,
                                vector<vector<double>> &bl_state_grid);

  vector<SearchOutcome> ComputeLocalSimilarity(
      const BoundaryData &boundary_data, ProfileParams &profile_params,
      SearchParams &search_params, vector<vector<double>> &bl_state_grid);

  vector<SearchOutcome> ComputeLocalSimilarityParallel(
      const BoundaryData &boundary_data, ProfileParams &profile_params,
      SearchParams &search_params, vector<vector<double>> &bl_state_grid,
      const int &nb_workers = 4);

  vector<SearchOutcome> ComputeDifferenceDifferential(
      const BoundaryData &boundary_data, ProfileParams &profile_params,
      SearchParams &search_params, vector<vector<double>> &bl_state_grid);

  // Post-processing
  vector<double> &GetEtaGrid(int worker_id = 0);
  vector<double> &GetStateGrid(int worker_id = 0);
  vector<double> &GetSensitivity(int worker_id = 0);

  void WriteEtaGrid(int worker_id = 0);
  void WriteStateGrid(const std::string &file_path, int profile_size,
                      int worker_id = 0);
  void WriteStateGrid(const std::string &file_path,
                      const vector<double> &state_grid, int profile_size);
  void WriteOutputGrid(const std::string &file_path,
                       const ProfileParams &profile_params, int profile_size,
                       int worker_id = 0);
  void WriteOutputGrid(const std::string &file_path,
                       const vector<double> &state_grid,
                       const vector<double> &eta_grid,
                       const ProfileParams &profile_params, int profile_size);

private:
  static constexpr size_t MAX_NB_WORKERS = 8;

  int _max_nb_steps;

  array<vector<double>, MAX_NB_WORKERS> state_grids;
  array<vector<double>, MAX_NB_WORKERS> eta_grids;
  array<vector<double>, MAX_NB_WORKERS> rhs_vecs;

  array<vector<double>, MAX_NB_WORKERS> sensitivity_matrices;
  array<vector<double>, 2 * MAX_NB_WORKERS> matrix_buffers;

  array<NewtonResources, MAX_NB_WORKERS> solver_resources;

  vector<double> field_grid;
  vector<double> output_grid;

  BLModel model_functions;
};

#endif