#ifndef BOUNDARY_LAYER_H
#define BOUNDARY_LAYER_H

#include "bl_model_struct.h"
#include "boundary_data_struct.h"
#include "profile_struct.h"
#include "search_struct.h"

#include <string>
#include <vector>

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
                              vector<double> &best_guess);
  SearchOutcome BoxProfileSearch(ProfileParams &profile_params,
                                 SearchParams &search_params,
                                 vector<double> &best_guess);
  SearchOutcome BoxProfileSearchParallel(ProfileParams &profile_params,
                                         SearchParams &search_params,
                                         vector<double> &best_guess);
  SearchOutcome
  BoxProfileSearchParallelWithQueues(ProfileParams &profile_params,
                                     SearchParams &search_params,
                                     vector<double> &best_guess);

  // Gradient Descent / Newton method
  SearchOutcome GradientProfileSearch(ProfileParams &profile_params,
                                      SearchParams &search_params,
                                      vector<double> &best_guess);

  // 2D profile calculation
  void Compute(const BoundaryData &boundary_data, ProfileParams &profile_params,
               SearchParams &search_params,
               vector<vector<double>> &bl_state_grid);
  void ComputeLocalSimilarity(const BoundaryData &boundary_data,
                              ProfileParams &profile_params,
                              SearchParams &search_params,
                              vector<vector<double>> &bl_state_grid);
  void ComputeDifferenceDifferential(const BoundaryData &boundary_data,
                                     ProfileParams &profile_params,
                                     SearchParams &search_params,
                                     vector<vector<double>> &bl_state_grid);

  // Post-processing
  vector<double> &GetEtaGrid(int worker_id = 0);
  vector<double> &GetStateGrid(int worker_id = 0);
  void WriteEtaGrid(int worker_id = 0);
  void WriteStateGrid(const std::string &file_path, int worker_id = 0);
  void WriteStateGrid(const std::string &file_path,
                      const vector<double> &state_grid);
  void WriteOutputGrid(const std::string &file_path,
                       const ProfileParams &profile_params, int profile_size,
                       int worker_id = 0);
  void WriteOutputGrid(const std::string &file_path,
                       const vector<double> &state_grid,
                       const vector<double> &eta_grid,
                       const ProfileParams &profile_params, int profile_size);

private:
  const int _max_nb_workers = 8;
  int _max_nb_steps;

  vector<vector<double>> state_grids;
  vector<vector<double>> eta_grids;
  vector<vector<double>> rhs_vecs;

  vector<vector<double>> sensitivity_matrices;
  vector<vector<double>> matrix_buffers;

  vector<double> field_grid;
  vector<double> output_grid;

  BLModel model_functions;
};

#endif