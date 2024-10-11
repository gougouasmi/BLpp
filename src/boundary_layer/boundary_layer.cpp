#include "boundary_layer.hpp"
#include "dense_direct_solver.hpp"
#include "dense_linalg.hpp"
#include "utils.hpp"

#include "profile_functions_default.hpp"
#include "score_functions.hpp"

#include <algorithm>
#include <cassert>
#include <tuple>

BoundaryLayer::BoundaryLayer(int max_nb_steps, BLModel model_functions)
    : _max_nb_steps(max_nb_steps), model_functions(model_functions),
      field_grid(FIELD_RANK * (max_nb_steps), 0.),
      output_grid(OUTPUT_RANK * (1 + _max_nb_steps)) {

  for (int worker_id = 0; worker_id < MAX_NB_WORKERS; worker_id++) {
    state_grids[worker_id].resize(BL_RANK * (1 + _max_nb_steps));
    eta_grids[worker_id].resize(1 + _max_nb_steps);
    rhs_vecs[worker_id].resize(BL_RANK);

    sensitivity_matrices[worker_id].resize(BL_RANK * 2);
    matrix_buffers[2 * worker_id + 0].resize(BL_RANK * BL_RANK);
    matrix_buffers[2 * worker_id + 1].resize(BL_RANK * BL_RANK);

    solver_resources[worker_id] = NewtonResources(BL_RANK);
  }
}

void BoundaryLayer::InitializeState(ProfileParams &profile_params,
                                    int worker_id) {
  // State grid
  if (profile_params.devel_mode != DevelMode::Tangent)
    model_functions.initialize(profile_params, state_grids[worker_id]);

  // Sensitivity matrix
  if (profile_params.devel_mode != DevelMode::Primal)
    model_functions.initialize_sensitivity(profile_params,
                                           sensitivity_matrices[worker_id]);
}

RhsFunction BoundaryLayer::GetRhsFun(SolveType solve_type) {
  if (solve_type == SolveType::SelfSimilar)
    return model_functions.compute_rhs_self_similar;

  if (solve_type == SolveType::LocallySimilar)
    return model_functions.compute_rhs_locally_similar;

  if (solve_type == SolveType::DifferenceDifferential)
    return model_functions.compute_rhs_diff_diff;

  return nullptr;
}

RhsJacobianFunction BoundaryLayer::GetJacobianFun(SolveType solve_type) {
  if (solve_type == SolveType::SelfSimilar)
    return model_functions.compute_rhs_jacobian_self_similar;

  if (solve_type == SolveType::LocallySimilar)
    return model_functions.compute_rhs_jacobian_locally_similar;

  if (solve_type == SolveType::DifferenceDifferential)
    return model_functions.compute_rhs_jacobian_diff_diff;

  return nullptr;
}

vector<double> &BoundaryLayer::GetEtaGrid(int worker_id) {
  return eta_grids[worker_id];
}

vector<double> &BoundaryLayer::GetStateGrid(int worker_id) {
  return state_grids[worker_id];
}

vector<double> &BoundaryLayer::GetSensitivity(int worker_id) {
  return sensitivity_matrices[worker_id];
}

void BoundaryLayer::WriteEtaGrid(int worker_id) {
  WriteH5("eta_grid.h5", GetEtaGrid(worker_id), "eta_grid");
}

void BoundaryLayer::WriteStateGrid(const std::string &file_path,
                                   int profile_size, int worker_id) {
  static vector<LabelIndex> state_labels = {
      {"C f''", FPP_ID},    {"f'", FP_ID}, {"f", F_ID},
      {"(C/Pr) g'", GP_ID}, {"g", G_ID},
  };

  WriteH5(file_path, state_grids[worker_id], state_labels, profile_size,
          BL_RANK, "state fields");
}

void BoundaryLayer::WriteStateGrid(const std::string &file_path,
                                   const vector<double> &state_grid,
                                   int profile_size) {
  static vector<LabelIndex> state_labels = {
      {"C f''", FPP_ID},    {"f'", FP_ID}, {"f", F_ID},
      {"(C/Pr) g'", GP_ID}, {"g", G_ID},
  };

  WriteH5(file_path, state_grid, state_labels, profile_size, BL_RANK,
          "state fields");
}

void BoundaryLayer::WriteOutputGrid(const std::string &file_path,
                                    const ProfileParams &profile_params,
                                    int profile_size, int worker_id) {
  // Compute outputs
  model_functions.compute_outputs(GetStateGrid(worker_id),
                                  GetEtaGrid(worker_id), output_grid,
                                  profile_size, profile_params);

  static vector<LabelIndex> output_labels = {
      {"ro", OUTPUT_RO_ID},      {"tau", OUTPUT_TAU_ID},
      {"q", OUTPUT_Q_ID},        {"y", OUTPUT_Y_ID},
      {"mu", OUTPUT_MU_ID},      {"C", OUTPUT_CHAPMANN_ID},
      {"Pr", OUTPUT_PRANDTL_ID},
  };

  WriteH5(file_path, output_grid, output_labels, profile_size, OUTPUT_RANK,
          "output fields");
}

void BoundaryLayer::WriteOutputGrid(const std::string &file_path,
                                    const vector<double> &state_grid,
                                    const vector<double> &eta_grid,
                                    const ProfileParams &profile_params,
                                    int profile_size) {
  // Compute outputs
  model_functions.compute_outputs(state_grid, eta_grid, output_grid,
                                  profile_size, profile_params);

  static vector<LabelIndex> output_labels = {
      {"ro", OUTPUT_RO_ID},      {"tau", OUTPUT_TAU_ID},
      {"q", OUTPUT_Q_ID},        {"y", OUTPUT_Y_ID},
      {"mu", OUTPUT_MU_ID},      {"C", OUTPUT_CHAPMANN_ID},
      {"Pr", OUTPUT_PRANDTL_ID},
  };

  WriteH5(file_path, output_grid, output_labels, profile_size, OUTPUT_RANK,
          "output fields");
}

int BoundaryLayer::DevelopProfile(ProfileParams &profile_params,
                                  vector<double> &score, int worker_id,
                                  bool advise) {
  int profile_size = -1;
  TimeScheme time_scheme = profile_params.scheme;

  // Clear state grid vector, unless you're going
  // to evolve the sensitivity
  if (profile_params.devel_mode != DevelMode::Tangent)
    std::fill(state_grids[worker_id].begin(), state_grids[worker_id].end(), 0.);

  if (time_scheme == TimeScheme::Explicit) {
    profile_size = DevelopProfileExplicit(profile_params, worker_id);
  } else if (time_scheme == TimeScheme::Implicit) {
    profile_size = DevelopProfileImplicit(profile_params, worker_id);
  } else if (time_scheme == TimeScheme::ImplicitCrankNicolson) {
    profile_size = DevelopProfileImplicitCN(profile_params, worker_id);
  } else {
    printf("Time Scheme not recognized!");
    return -1;
  }

  // Terminate if you couldn't even move forward one step
  if (profile_size < 0) {
    return profile_size;
  }

  ////
  // Compute score, score_jacobian, and advise if requested
  //
  int state_offset = BL_RANK * profile_size;

  vector<double> &state_grid = state_grids[worker_id];

  ComputeScore(profile_params, state_grid, state_offset, score);

  if (advise) {
    vector<double> &sensitivity = sensitivity_matrices[worker_id];

    printf("Sensitivity matrix:\n");
    utils::print_matrix_column_major(sensitivity, BL_RANK, 2);

    vector<double> score_jacobian(4, 0.); // row_major

    ComputeScoreJacobian(profile_params, state_grid, state_offset, sensitivity,
                         0, score_jacobian);

    printf("Score jacobian:\n");
    utils::print_matrix_row_major(score_jacobian, 2, 2, 2);

    vector<double> delta(2, 0);
    delta[0] = -score[0];
    delta[1] = -score[1];

    pair<vector<double>, vector<double>> &lu_resources =
        solver_resources[worker_id].matrix.GetLU();
    LUSolve(score_jacobian, delta, 2, lu_resources);

    printf("Delta suggestion:\n");
    utils::print_matrix_column_major(delta, 2, 1);
  }

  return profile_size;
}

int BoundaryLayer::DevelopProfileExplicit(ProfileParams &profile_params,
                                          int worker_id) {
  assert(profile_params.AreValid());

  const DevelMode devel_mode = profile_params.devel_mode;

  RhsFunction compute_rhs = GetRhsFun(profile_params.solve_type);
  assert(compute_rhs != nullptr);

  vector<double> &state_grid = state_grids[worker_id];
  vector<double> &eta_grid = eta_grids[worker_id];
  vector<double> &rhs = rhs_vecs[worker_id];
  vector<double> &sensitivity = sensitivity_matrices[worker_id];

  const int grid_nb_steps = eta_grid.size() - 1;
  const int max_nb_steps = profile_params.nb_steps;

  assert(max_nb_steps > 1);
  assert(max_nb_steps <= grid_nb_steps);

  InitializeState(profile_params, worker_id);
  double max_step = profile_params.max_step;

  // printf("Sensitivity matrix after initialize:\n");
  // print_matrix_column_major(sensitivity, BL_RANK, 2);

  vector<double> &jacobian_buffer_rm = matrix_buffers[2 * worker_id + 0];
  vector<double> &matrix_buffer_cm = matrix_buffers[2 * worker_id + 1];

  RhsJacobianFunction compute_rhs_jacobian =
      GetJacobianFun(profile_params.solve_type);
  assert(compute_rhs_jacobian != nullptr);

  //
  int step_id = 0;
  int state_offset = 0;
  int field_offset = 0;
  while (step_id < max_nb_steps) {

    // Evolve state forward
    double eta_step =
        std::min(max_step, compute_rhs(state_grid, state_offset, field_grid,
                                       field_offset, rhs, profile_params));

    // Evolve state/grid forward
    eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;
    for (int var_id = 0; var_id < BL_RANK; ++var_id) {
      state_grid[state_offset + BL_RANK + var_id] =
          state_grid[state_offset + var_id] + eta_step * rhs[var_id];
    }

    if (devel_mode == DevelMode::Full) {
      // Evolve sensitivity forward: S^{n+1} = (I + dt * J^{n}) S^{n}
      compute_rhs_jacobian(state_grid, state_offset, field_grid, field_offset,
                           jacobian_buffer_rm, profile_params);
      int buf_offset = 0;
      for (int row_id = 0; row_id < BL_RANK; row_id++) {
        for (int col_id = 0; col_id < row_id; col_id++) {
          jacobian_buffer_rm[buf_offset + col_id] *= eta_step;
        }

        jacobian_buffer_rm[buf_offset + row_id] =
            1. + eta_step * jacobian_buffer_rm[buf_offset + row_id];

        for (int col_id = row_id + 1; col_id < BL_RANK; col_id++) {
          jacobian_buffer_rm[buf_offset + col_id] *= eta_step;
        }

        buf_offset += BL_RANK;
      }

      std::copy(sensitivity.begin(), sensitivity.end(),
                matrix_buffer_cm.begin());
      DenseMatrixMatrixMultiply(jacobian_buffer_rm, matrix_buffer_cm,
                                sensitivity, BL_RANK, 2);
    }

    // Debugging
    if (isnan(state_grid[state_offset + BL_RANK + G_ID])) {
      printf("g^{%d+1} = nan!, step = %.2e, rhs[G_ID] = %.2e, state[G_ID] = "
             "%.2e.\n",
             step_id, eta_step, rhs[G_ID], state_grid[state_offset + G_ID]);
      utils::print_state(rhs, 0, BL_RANK);
      utils::print_state(state_grid, state_offset, BL_RANK);
      assert(false);
    }

    if ((state_grid[state_offset + BL_RANK + G_ID]) < 0) {
      printf(
          "g^{%d+1} < 0! step = %.2e, rhs[G_ID] = %.2e, state[G_ID] = %.2e.\n",
          step_id, eta_step, rhs[G_ID], state_grid[state_offset + G_ID]);
      utils::print_state(rhs, 0, BL_RANK);
      utils::print_state(state_grid, state_offset, BL_RANK);
      assert(false);
    }

    // Update indexing
    state_offset += BL_RANK;
    field_offset += FIELD_RANK;
    step_id += 1;
  }

  return step_id;
}

int BoundaryLayer::DevelopProfileImplicit(ProfileParams &profile_params,
                                          int worker_id) {
  assert(profile_params.AreValid());

  const DevelMode devel_mode = profile_params.devel_mode;

  RhsFunction compute_rhs = GetRhsFun(profile_params.solve_type);
  RhsJacobianFunction compute_rhs_jacobian =
      GetJacobianFun(profile_params.solve_type);

  assert(compute_rhs != nullptr);
  assert(compute_rhs_jacobian != nullptr);

  vector<double> &state_grid = state_grids[worker_id];
  vector<double> &eta_grid = eta_grids[worker_id];
  vector<double> &rhs = rhs_vecs[worker_id];
  vector<double> &sensitivity = sensitivity_matrices[worker_id];

  const int grid_nb_steps = eta_grid.size() - 1;
  const int max_nb_steps = profile_params.nb_steps;

  assert(max_nb_steps > 1);
  assert(max_nb_steps <= grid_nb_steps);

  InitializeState(profile_params, worker_id);
  double eta_step = profile_params.max_step;

  //
  int step_id = 0;
  int state_offset = 0;
  int field_offset = 0;

  ////
  // Setup nonlinear solver functions
  //
  int xdim = BL_RANK;

  // (1 / 3) Objective function
  auto objective_fun = [xdim, worker_id, &eta_step, &state_offset,
                        &field_offset, &profile_params, &state_grid,
                        compute_rhs, this](const vector<double> &state,
                                           vector<double> &residual) {
    // U^{n+1} - U^{n} - step * R(U^{n+1}) = 0
    compute_rhs(state, 0, field_grid, field_offset, residual, profile_params);
    for (int idx = 0; idx < xdim; idx++) {
      residual[idx] *= -eta_step;
      residual[idx] += state[idx] - state_grid[state_offset + idx];
    };
  };

  // (2 / 3) Limit update function
  auto limit_update_fun = [xdim, &profile_params,
                           this](const vector<double> &state,
                                 const vector<double> &state_varn) {
    return model_functions.limit_update(state, state_varn, profile_params);
  };

  // (3 / 3) Jacobian function
  auto jacobian_fun = [xdim, &eta_step, &field_offset, &profile_params,
                       compute_rhs_jacobian,
                       this](const vector<double> &state, DenseMatrix &matrix) {
    vector<double> &matrix_data = matrix.GetData();

    compute_rhs_jacobian(state, 0, field_grid, field_offset, matrix_data,
                         profile_params);
    int local_offset = 0;
    for (int idx = 0; idx < xdim; idx++) {

      for (int idy = 0; idy < idx; idy++) {
        matrix_data[local_offset] *= -eta_step;
        local_offset += 1;
      }

      matrix_data[local_offset] *= -eta_step;
      matrix_data[local_offset] += 1.;
      local_offset += 1;

      for (int idy = idx + 1; idy < xdim; idy++) {
        matrix_data[local_offset] *= -eta_step;
        local_offset += 1;
      }
    }
  };

  NewtonParams newton_params;
  NewtonResources &newton_resources = solver_resources[worker_id];

  vector<double> &solution_buffer = newton_resources.state;
  std::copy(state_grid.begin(), state_grid.begin() + xdim,
            solution_buffer.begin());

  ////
  // Time loop
  //

  while (step_id < max_nb_steps) {

    // Evolve state forward by solving the nonlinear system
    if (devel_mode != DevelMode::Tangent) {

      bool pass =
          NewtonSolveDirect(solution_buffer, objective_fun, jacobian_fun,
                            limit_update_fun, newton_params, newton_resources);
      if (!pass) {
        break;
      }

      // Evolve state/grid forward
      eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;

      for (int idx = 0; idx < xdim; idx++) {
        state_grid[state_offset + BL_RANK + idx] = solution_buffer[idx];
      }
    } else { // It is a tangent solve

      for (int idx = 0; idx < xdim; idx++) {
        solution_buffer[idx] = state_grid[state_offset + BL_RANK + idx];
      }
      jacobian_fun(solution_buffer, newton_resources.matrix);
    }

    if (devel_mode != DevelMode::Primal) {

      // Evolve sensitivity forward: (I - delta * J^{n+1}) S^{n+1} = S^{n}
      //  -> The Jacobian (I - delta * J^{n+1}) is located within
      //  newton_resources
      newton_resources.matrix.MatrixSolve(sensitivity, BL_RANK, 2);
    }

    // Update indexing
    state_offset += BL_RANK;
    field_offset += FIELD_RANK;

    step_id += 1;
  }

  return step_id;
}

int BoundaryLayer::DevelopProfileImplicitCN(ProfileParams &profile_params,
                                            int worker_id) {
  assert(profile_params.AreValid());

  const DevelMode devel_mode = profile_params.devel_mode;

  RhsFunction compute_rhs = GetRhsFun(profile_params.solve_type);
  RhsJacobianFunction compute_rhs_jacobian =
      GetJacobianFun(profile_params.solve_type);

  assert(compute_rhs != nullptr);
  assert(compute_rhs_jacobian != nullptr);

  vector<double> &state_grid = state_grids[worker_id];
  vector<double> &eta_grid = eta_grids[worker_id];
  vector<double> &rhs = rhs_vecs[worker_id];
  vector<double> &sensitivity = sensitivity_matrices[worker_id];

  vector<double> &jacobian_buffer_rm = matrix_buffers[2 * worker_id + 0];
  vector<double> &matrix_buffer_cm = matrix_buffers[2 * worker_id + 1];

  const int grid_nb_steps = eta_grid.size() - 1;
  const int max_nb_steps = profile_params.nb_steps;

  assert(max_nb_steps > 1);
  assert(max_nb_steps <= grid_nb_steps);

  InitializeState(profile_params, worker_id);
  double eta_step = profile_params.max_step;

  //
  int step_id = 0;
  int state_offset = 0;
  int field_offset = 0;

  ////
  // Setup nonlinear solver functions
  //
  int xdim = BL_RANK;

  // (1 / 3) Objective function
  //   -> R(U^{n}) pre-computed and found in rhs buffer
  auto objective_fun = [xdim, worker_id, &eta_step, &state_offset,
                        &field_offset, &profile_params, &state_grid, &rhs,
                        compute_rhs, this](const vector<double> &state,
                                           vector<double> &residual) {
    // U^{n+1} - U^{n} - 0.5 * step * (R(U^{n} + R(U^{n+1}) = 0
    compute_rhs(state, 0, field_grid, field_offset, residual, profile_params);

    for (int idx = 0; idx < xdim; idx++) {
      residual[idx] = (state[idx] - state_grid[state_offset + idx]) -
                      0.5 * eta_step * (rhs[idx] + residual[idx]);
    };
  };

  // (2 / 3) Limit update function
  auto limit_update_fun = [xdim, &profile_params,
                           this](const vector<double> &state,
                                 const vector<double> &state_varn) {
    return model_functions.limit_update(state, state_varn, profile_params);
  };

  // (3 / 3) Jacobian function
  auto jacobian_fun = [xdim, &eta_step, &state_offset, &field_offset,
                       &profile_params, compute_rhs_jacobian,
                       this](const vector<double> &state, DenseMatrix &matrix) {
    // (I - 0.5 * eta_step * J(X))
    vector<double> &matrix_data = matrix.GetData();

    compute_rhs_jacobian(state, 0, field_grid, field_offset, matrix_data,
                         profile_params);
    int local_offset = 0;
    for (int idx = 0; idx < xdim; idx++) {

      for (int idy = 0; idy < idx; idy++) {
        matrix_data[local_offset] *= -0.5 * eta_step;
        local_offset += 1;
      }

      matrix_data[local_offset] =
          1 - 0.5 * eta_step * matrix_data[local_offset];
      local_offset += 1;

      for (int idy = idx + 1; idy < xdim; idy++) {
        matrix_data[local_offset] *= -0.5 * eta_step;
        local_offset += 1;
      }
    }
  };

  NewtonParams newton_params;
  NewtonResources &newton_resources = solver_resources[worker_id];
  vector<double> &solution_buffer = newton_resources.state;

  for (int idx = 0; idx < xdim; idx++) {
    solution_buffer[idx] = state_grid[idx];
  }

  // Time loop
  while (step_id < max_nb_steps) {

    // Prepare data for nonlinear solve
    //   - Pre-compute R(U^{n})
    compute_rhs(state_grid, state_offset, field_grid, field_offset, rhs,
                profile_params);

    // Evolve state forward by solving the nonlinear system
    bool pass =
        NewtonSolveDirect(solution_buffer, objective_fun, jacobian_fun,
                          limit_update_fun, newton_params, newton_resources);
    if (!pass) {
      // printf("unsuccessful solve. Score = [%.2e, %.2e]\n", score[0],
      // score[1]);
      break;
    }

    // Evolve state/grid forward
    eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;

    for (int var_id = 0; var_id < BL_RANK; ++var_id) {
      state_grid[state_offset + BL_RANK + var_id] = solution_buffer[var_id];
    }

    if (devel_mode == DevelMode::Full) {
      // Evolve sensitivity forward
      // dS/dt = J * S,
      // -> S^{n+1} - S^{n} = 0.5 * step * (J^{n+1} S^{n+1} + J^{n} S^{n})
      // -> (I - 0.5 * step * J^{n+1}) S^{n+1} = (I + 0.5 * step * J^{n}) S^{n}

      //    (1 / 3) Evaluate right-hand side (matrix-matrix product)
      compute_rhs_jacobian(state_grid, state_offset, field_grid, field_offset,
                           jacobian_buffer_rm, profile_params);
      int offset = 0;
      for (int idx = 0; idx < xdim; idx++) {
        for (int idy = 0; idy < idx; idy++) {
          jacobian_buffer_rm[offset + idy] *= 0.5 * eta_step;
        }

        jacobian_buffer_rm[offset + idx] =
            1. + 0.5 * eta_step * jacobian_buffer_rm[offset + idx];

        for (int idy = idx + 1; idy < xdim; idy++) {
          jacobian_buffer_rm[offset + idy] *= 0.5 * eta_step;
        }
        offset += BL_RANK;
      }
      std::copy(sensitivity.begin(), sensitivity.end(),
                matrix_buffer_cm.begin());
      DenseMatrixMatrixMultiply(jacobian_buffer_rm, matrix_buffer_cm,
                                sensitivity, BL_RANK, 2);

      //    (2 / 3) Left-hand side matrix is the residual Jacobian in the Newton
      //    solve

      //    (3 / 3) Solve for next sensitivity
      newton_resources.matrix.MatrixSolve(sensitivity, BL_RANK, 2);
    }

    // Update indexing
    state_offset += BL_RANK;
    field_offset += FIELD_RANK;

    step_id += 1;
  }

  return step_id;
}

/////
// Search methods
//

SearchOutcome BoundaryLayer::ProfileSearch(ProfileParams &profile_params,
                                           SearchParams &search_params,
                                           array<double, 2> &best_guess) {
  SearchMethod method = search_params.method;

  if (method == SearchMethod::BoxSerial) {
    return BoxProfileSearch(profile_params, search_params, best_guess);
  }

  if (method == SearchMethod::BoxParallel) {
    return BoxProfileSearchParallel(profile_params, search_params, best_guess);
  }

  if (method == SearchMethod::BoxParallelQueue) {
    return BoxProfileSearchParallelWithQueues(profile_params, search_params,
                                              best_guess);
  }

  if (method == SearchMethod::GradientSerial) {
    return GradientProfileSearch(profile_params, search_params, best_guess);
  }

  return SearchOutcome{false, -1, 1};
}

SearchOutcome BoundaryLayer::BoxProfileSearch(ProfileParams &profile_params,
                                              SearchParams &search_params,
                                              array<double, 2> &best_guess) {
  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  profile_params.scoring = search_params.scoring;
  profile_params.devel_mode = DevelMode::Primal;

  // Fetch search window parameters
  SearchWindow &window = search_params.window;
  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  if (verbose) {
    printf("Iter 0 - window "
           "=[[%.2e, %.2e], [%.2e, %.2e]\n",
           fpp_min, fpp_max, gp_min, gp_max);
  }

  const int worker_id = 0;

  // Temporary arrays
  array<double, 2> initial_guess{{0.0, 0.0}};
  vector<double> score(2, 0.0);

  double res_norm;
  double min_res_norm;

  int best_profile_size = 1;

  for (int iter = 0; iter < max_iter; iter++) {

    double delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    double delta_gp = (gp_max - gp_min) / (ydim - 1);

    double fpp0 = fpp_min;
    double gp0 = gp_min;

    min_res_norm = 1e30;

    int min_fid = 0;
    int min_gid = 0;

    // (1 / 2) Develop profiles on square grid of initial conditions
    for (int fid = 0; fid < xdim; fid++) {
      initial_guess[0] = fpp0;

      gp0 = gp_min;
      for (int gid = 0; gid < ydim; gid++) {
        initial_guess[1] = gp0;

        profile_params.SetInitialValues(initial_guess);
        int profile_size = DevelopProfile(profile_params, score);

        if (profile_size > 100) {
          res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

          if (res_norm < min_res_norm) {

            min_res_norm = res_norm;
            best_profile_size = profile_size;
            min_fid = fid;
            min_gid = gid;

            if (res_norm < rtol) {
              if (verbose) {
                printf("Solution found: f''(0)=%.2e, g'(0)=%.2e.\n", fpp0, gp0);
              }
              best_guess[0] = fpp0;
              best_guess[1] = gp0;
              return SearchOutcome{true, worker_id, profile_size, best_guess};
            }
          }
        }

        gp0 += delta_gp;
      }

      fpp0 += delta_fpp;
    }

    if (min_res_norm == 1e30) {
      printf("STOP CONDITION: Not a single converged profile, aborting.\n\n");
      return SearchOutcome{false, worker_id, 1, best_guess};
    }

    // (2 / 2) Define next bounds
    double x0 = fpp_min, x1 = fpp_max;
    double xs = fpp_min + delta_fpp * min_fid;

    if (min_fid == 0) {
      fpp_min = 0.5 * x0;
      fpp_max = 0.5 * (x0 + x1);
    } else {
      if (min_fid == xdim - 1) {
        fpp_max += 0.5 * (x1 - x0);
        fpp_min = 0.5 * (x0 + x1);
      } else {
        fpp_min = 0.5 * (x0 + xs);
        fpp_max = 0.5 * (xs + x1);
      }
    }

    double y0 = gp_min, y1 = gp_max;
    double ys = gp_min + delta_gp * min_gid;

    if (min_gid == 0) {
      gp_min = 0.5 * y0;
      gp_max = 0.5 * (y0 + y1);
    } else {
      if (min_gid == ydim - 1) {
        gp_max += 0.5 * (y1 - y0);
        gp_min = 0.5 * (y0 + y1);
      } else {
        gp_min = 0.5 * (y0 + ys);
        gp_max = 0.5 * (ys + y1);
      }
    }

    if (verbose) {
      printf("Iter %d - score=%.2e, f''(0)=%.2e, g'(0)=%.2e, next "
             "window "
             "=[[%.2e, %.2e], [%.2e, %.2e]\n",
             iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
    }
  }

  return SearchOutcome{min_res_norm < rtol, worker_id, best_profile_size,
                       best_guess};
}

#include <future>
#include <thread>

SearchOutcome
BoundaryLayer::BoxProfileSearchParallel(ProfileParams &profile_params,
                                        SearchParams &search_params,
                                        array<double, 2> &best_guess) {
  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  profile_params.scoring = search_params.scoring;
  profile_params.devel_mode = DevelMode::Primal;

  // Fetch search window parameters
  SearchWindow &window = search_params.window;
  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  double min_res_norm = 1e30;
  int best_worker_id = -1;
  int best_profile_size = 1;

  double delta_fpp, delta_gp;

  // Define local search
  auto local_search_task =
      [&fpp_min, &gp_min, &delta_fpp, &delta_gp, rtol, profile_params,
       this](const int fid_start, const int fid_end, const int gid_start,
             const int gid_end, const int worker_id) mutable {
        array<double, 2> initial_guess{{0.0, 0.0}};
        vector<double> score(2, 0.0);

        double res_norm;
        double min_res_norm = 1e30;

        int best_profile_size = 1;

        int min_fid = fid_start, min_gid = gid_start;

        double fpp0 = fpp_min + fid_start * delta_fpp;
        for (int fid = fid_start; fid < fid_end; fid++) {
          initial_guess[0] = fpp0;

          double gp0 = gp_min + gid_start * delta_gp;
          for (int gid = gid_start; gid < gid_end; gid++) {
            initial_guess[1] = gp0;

            profile_params.SetInitialValues(initial_guess);
            int profile_size = DevelopProfile(profile_params, score, worker_id);

            if (profile_size > 100) {
              res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

              if (res_norm < min_res_norm) {

                min_res_norm = res_norm;
                best_profile_size = profile_size;

                min_fid = fid;
                min_gid = gid;

                if (res_norm < rtol) {
                  return BoxSearchResult(min_res_norm, min_fid, min_gid,
                                         worker_id, best_profile_size);
                }
              }
            }

            gp0 += delta_gp;
          }

          fpp0 += delta_fpp;
        }
        return BoxSearchResult(min_res_norm, min_fid, min_gid, worker_id,
                               best_profile_size);
      };

  for (int iter = 0; iter < max_iter; iter++) {

    delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    delta_gp = (gp_max - gp_min) / (ydim - 1);

    // (1 / 2) Develop profiles on square grids
    vector<std::future<BoxSearchResult>> futures;

    futures.emplace_back(std::async(std::launch::async, local_search_task, 0,
                                    xdim / 2, 0, ydim / 2, 0));
    futures.emplace_back(std::async(std::launch::async, local_search_task,
                                    xdim / 2, xdim, 0, ydim / 2, 1));
    futures.emplace_back(std::async(std::launch::async, local_search_task, 0,
                                    xdim / 2, ydim / 2, ydim, 2));
    futures.emplace_back(std::async(std::launch::async, local_search_task,
                                    xdim / 2, xdim, ydim / 2, ydim, 3));

    min_res_norm = 1e30;
    int min_fid = 0, min_gid = 0;

    for (auto &ftr : futures) {
      BoxSearchResult result = ftr.get();
      double res_norm = result.res_norm;
      if (res_norm < min_res_norm) {
        min_res_norm = res_norm;
        min_fid = result.xid;
        min_gid = result.yid;
        best_worker_id = result.worker_id;
        best_profile_size = result.profile_size;
      }
    }

    // If your best guess is good enough, terminate.
    if (min_res_norm < rtol) {
      best_guess[0] = fpp_min + min_fid * delta_fpp;
      best_guess[1] = gp_min + min_gid * delta_gp;

      if (verbose) {
        printf("Solution found: f''(0)=%.2e, g'(0)=%.2e.\n", best_guess[0],
               best_guess[1]);
      }

      return SearchOutcome{true, best_worker_id, best_profile_size, best_guess};
    }

    // (2 / 2) Define next bounds
    double x0 = fpp_min, x1 = fpp_max;
    double xs = fpp_min + delta_fpp * min_fid;

    if (min_fid == 0) {
      fpp_min = 0.5 * x0;
      fpp_max = 0.5 * (x0 + x1);
    } else {
      if (min_fid == xdim - 1) {
        fpp_max += 0.5 * (x1 - x0);
        fpp_min = 0.5 * (x0 + x1);
      } else {
        fpp_min = 0.5 * (x0 + xs);
        fpp_max = 0.5 * (xs + x1);
      }
    }

    double y0 = gp_min, y1 = gp_max;
    double ys = gp_min + delta_gp * min_gid;

    if (min_gid == 0) {
      gp_min = 0.5 * y0;
      // gp_min -= 0.5 * (y1 - y0);
      gp_max = 0.5 * (y0 + y1);
    } else {
      if (min_gid == ydim - 1) {
        gp_max += 0.5 * (y1 - y0);
        gp_min = 0.5 * (y0 + y1);
      } else {
        gp_min = 0.5 * (y0 + ys);
        gp_max = 0.5 * (ys + y1);
      }
    }

    if (verbose) {
      printf("Iter %d - score=%.2e, f''(0)=%.2e, g'(0)=%.2e, next "
             "window "
             "=[[%.2e, %.2e], [%.2e, %.2e]\n",
             iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
    }
  }

  return SearchOutcome{min_res_norm < rtol, best_worker_id, best_profile_size,
                       best_guess};
}

#include "message_queue.hpp"
#include <memory>

SearchOutcome BoundaryLayer::BoxProfileSearchParallelWithQueues(
    ProfileParams &profile_params, SearchParams &search_params,
    array<double, 2> &best_guess) {
  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  profile_params.scoring = search_params.scoring;
  profile_params.devel_mode = DevelMode::Primal;

  // Fetch search window parameters
  SearchWindow &window = search_params.window;
  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  int best_worker_id = -1;
  int best_profile_size = 1;

  // Define queues for BoxSearchInput and BoxSearchResult
  std::shared_ptr<MessageQueue<BoxSearchInput>> work_queue(
      new MessageQueue<BoxSearchInput>());
  std::shared_ptr<MessageQueue<BoxSearchResult>> result_queue(
      new MessageQueue<BoxSearchResult>());

  // Spawn worker threads
  auto local_search_task = [rtol, profile_params, work_queue, result_queue,
                            this](const int worker_id) mutable {
    while (true) {
      BoxSearchInput inputs = work_queue->fetch();

      if (inputs.Stop()) {
        break;
      }

      bool result_sent = false;

      // Search!
      const double fpp_min = inputs.x0;
      const double delta_fpp = inputs.dx;
      const double gp_min = inputs.y0;
      const double delta_gp = inputs.dy;

      const int fid_start = inputs.xid_start;
      const int fid_end = inputs.xid_end;
      const int gid_start = inputs.yid_start;
      const int gid_end = inputs.yid_end;

      array<double, 2> initial_guess{{0.0, 0.0}};
      vector<double> score(2, 0.0);

      double res_norm;
      double min_res_norm = 1e30;

      int best_profile_size = 1;

      int min_fid = fid_start, min_gid = gid_start;

      double fpp0 = fpp_min + fid_start * delta_fpp;
      for (int fid = fid_start; fid < fid_end; fid++) {
        initial_guess[0] = fpp0;

        double gp0 = gp_min + gid_start * delta_gp;
        for (int gid = gid_start; gid < gid_end; gid++) {
          initial_guess[1] = gp0;

          profile_params.SetInitialValues(initial_guess);
          int profile_size = DevelopProfile(profile_params, score, worker_id);

          if (profile_size > 100) {
            res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

            if (res_norm < min_res_norm) {

              min_res_norm = res_norm;
              min_fid = fid;
              min_gid = gid;
              best_profile_size = profile_size;

              if (res_norm < rtol) {
                result_queue->send(BoxSearchResult(
                    min_res_norm, min_fid, min_gid, worker_id, profile_size));

                // Force nested loop termination
                fid = fid_end;
                gid = gid_end;
                result_sent = true;
              }
            }
          }

          gp0 += delta_gp;
        }

        fpp0 += delta_fpp;
      }

      if (!result_sent) {
        result_queue->send(BoxSearchResult(min_res_norm, min_fid, min_gid,
                                           worker_id, best_profile_size));
      }
    }
  };

  vector<std::future<void>> futures;
  for (int worker_id = 0; worker_id < 4; worker_id++) {
    futures.emplace_back(
        std::async(std::launch::async, local_search_task, worker_id));
  }

  double min_res_norm;
  for (int iter = 0; iter < max_iter; iter++) {

    // Compute step sizes
    double delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    double delta_gp = (gp_max - gp_min) / (ydim - 1);

    // Send work orders
    work_queue->send(BoxSearchInput(fpp_min, delta_fpp, gp_min, delta_gp, 0,
                                    xdim / 2, 0, ydim / 2));
    work_queue->send(BoxSearchInput(fpp_min, delta_fpp, gp_min, delta_gp,
                                    xdim / 2, xdim, 0, ydim / 2));
    work_queue->send(BoxSearchInput(fpp_min, delta_fpp, gp_min, delta_gp, 0,
                                    xdim / 2, ydim / 2, ydim));
    work_queue->send(BoxSearchInput(fpp_min, delta_fpp, gp_min, delta_gp,
                                    xdim / 2, xdim, ydim / 2, ydim));

    // Fetch best guesses from workers
    min_res_norm = 1e30;
    int min_fid = 0, min_gid = 0;

    int processed_results = 0;
    while (processed_results < 4) {
      BoxSearchResult result = result_queue->fetch();

      if (result.res_norm < min_res_norm) {
        min_res_norm = result.res_norm;
        min_fid = result.xid;
        min_gid = result.yid;
        best_worker_id = result.worker_id;
        best_profile_size = result.profile_size;
      }

      processed_results += 1;
    }

    // If your best guess is good enough, terminate.
    if (min_res_norm < rtol) {
      best_guess[0] = fpp_min + min_fid * delta_fpp;
      best_guess[1] = gp_min + min_gid * delta_gp;
      if (verbose) {
        printf("Solution found: f''(0)=%.2e, g'(0)=%.2e.\n", best_guess[0],
               best_guess[1]);
      }
      work_queue->stop();
      return SearchOutcome{true, best_worker_id, best_profile_size, best_guess};
    }

    // (2 / 2) Define next bounds
    double x0 = fpp_min, x1 = fpp_max;
    double xs = fpp_min + delta_fpp * min_fid;

    if (min_fid == 0) {
      fpp_min = 0.5 * x0;
      fpp_max = 0.5 * (x0 + x1);
    } else {
      if (min_fid == xdim - 1) {
        fpp_max += 0.5 * (x1 - x0);
        fpp_min = 0.5 * (x0 + x1);
      } else {
        fpp_min = 0.5 * (x0 + xs);
        fpp_max = 0.5 * (xs + x1);
      }
    }

    double y0 = gp_min, y1 = gp_max;
    double ys = gp_min + delta_gp * min_gid;

    if (min_gid == 0) {
      gp_min = 0.5 * y0;
      gp_max = 0.5 * (y0 + y1);
    } else {
      if (min_gid == ydim - 1) {
        gp_max += 0.5 * (y1 - y0);
        gp_min = 0.5 * (y0 + y1);
      } else {
        gp_min = 0.5 * (y0 + ys);
        gp_max = 0.5 * (ys + y1);
      }
    }

    if (verbose) {
      printf("Iter %d - score=%.2e, f''(0)=%.2e, g'(0)=%.2e, next "
             "window "
             "=[[%.2e, %.2e], [%.2e, %.2e]\n",
             iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
    }
  }

  work_queue->stop();

  std::for_each(futures.begin(), futures.end(),
                [](std::future<void> &ftr) { ftr.wait(); });

  return SearchOutcome{min_res_norm < rtol, best_worker_id, best_profile_size,
                       best_guess};
}

SearchOutcome BoundaryLayer::GradientProfileSearch(
    ProfileParams &profile_params, SearchParams &search_params,
    array<double, 2> &best_guess, int worker_id) {
  assert(array_norm<2>(best_guess) > 0);

  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  profile_params.scoring = search_params.scoring;
  profile_params.devel_mode = DevelMode::Full;

  // Get worker id to fetch appropriate resources
  vector<double> &sensitivity = sensitivity_matrices[worker_id];

  vector<double> score(2, 0);
  vector<double> score_jacobian(4, 0.); // row_major
  vector<double> delta(2, 0);

  pair<vector<double>, vector<double>> &lu_resources =
      solver_resources[worker_id].matrix.GetLU();

  double snorm;
  int best_profile_size = 1;

  // Profiling //
  static int devel_count = 0;
  static int devel_with_grad_count = 0;

  // Compute initial profile
  profile_params.SetInitialValues(best_guess);
  int profile_size = DevelopProfile(profile_params, score, worker_id);

  devel_count++;

  snorm = vector_norm(score);

  if (verbose)
    printf("Iter #0, f''(0)=%.5e, g'(0)=%.5e, ||e||=%.5e.\n", best_guess[0],
           best_guess[1], snorm);

  if (snorm < rtol) {
    if (verbose)
      printf("  -> Successful search.\n");
    return SearchOutcome{true, worker_id, profile_size, best_guess};
  }

  // Start main loop
  array<double, 2> guess = {{best_guess[0], best_guess[1]}};

  int iter = 0;
  while (iter < max_iter) {

    devel_with_grad_count++;

    // Assemble score jacobian
    ComputeScoreJacobian(profile_params, state_grids[worker_id],
                         profile_size * BL_RANK, sensitivity, 0,
                         score_jacobian);

    if (verbose) {
      printf("\nIter #%d - START\n", iter + 1);
      printf(" -> Score: \n");
      utils::print_state(score, 0, 2, 2);
      printf(" -> State sentivity: \n");
      utils::print_matrix_column_major(sensitivity, BL_RANK, 2, 2);
      printf(" -> Score Jacobian:\n");
      utils::print_matrix_row_major(score_jacobian, 2, 2, 2);
    }

    // Solve linear system
    delta[0] = -score[0];
    delta[1] = -score[1];

    LUSolve(score_jacobian, delta, 2, lu_resources);

    if (isnan(delta[0]) || isnan(delta[1]))
      return SearchOutcome{false, worker_id, 1, best_guess};

    if (verbose) {
      printf(" -> delta: ");
      utils::print_state(delta, 0, 2);
      printf(" -> Score Jacobian determinant: %.3e.\n",
             UpperDeterminant(lu_resources.second, 2));
      printf("\n");
    }

    // Line search
    //    (1 / 2) : Lower coeff to meet conditions
    double alpha = 1;
    if (delta[0] != 0) {
      alpha = std::min(alpha, 0.5 * fabs(guess[0] / delta[0]));
    }

    if (delta[1] != 0) {
      alpha = std::min(alpha, 0.5 * fabs(guess[1] / delta[1]));
    }

    //    (2 / 2) : Lower alpha until score drops
    guess[0] += alpha * delta[0];
    guess[1] += alpha * delta[1];

    profile_params.SetInitialValues(guess);
    int profile_size = DevelopProfile(profile_params, score, worker_id);
    assert(profile_size > 20);

    devel_count++;

    double ls_norm = vector_norm(score);
    bool ls_pass = ls_norm < snorm;

    if (!ls_pass) {
      for (int ls_iter = 0; ls_iter < 20; ls_iter++) {

        alpha *= 0.5;

        guess[0] -= alpha * delta[0];
        guess[1] -= alpha * delta[1];

        profile_params.SetInitialValues(guess);
        int profile_size = DevelopProfile(profile_params, score, worker_id);
        assert(profile_size > 20);

        devel_count++;

        ls_norm = vector_norm(score);

        if (verbose) {
          printf("   LS Iter #%d, score_norm: %.2e, score: ", ls_iter + 1,
                 ls_norm);
          utils::print_state(score, 0, 2);
        }

        if (ls_norm < snorm) {
          snorm = ls_norm;
          ls_pass = true;
          best_profile_size = profile_size;
          break;
        }
      }
    }

    if (ls_pass) {
      snorm = ls_norm;
      best_profile_size = profile_size;

      // Save progress to best_guess
      std::copy(guess.begin(), guess.end(), best_guess.begin());

      if (verbose)
        printf("Iter #%d, f''(0)=%.5e, g'(0)=%.5e, ||e||=%.5e, ||delta||=%.5e, "
               "alpha=%.2e.\n",
               iter + 1, guess[0], guess[1], snorm, vector_norm(delta), alpha);

      if (snorm < rtol) {
        if (verbose) {
          printf("  -> Successful search.\n");
          if (profile_size < _max_nb_steps) {
            printf("   -> Yet the run did not complete! (%d / %d)\n",
                   profile_size, _max_nb_steps);
            printf("   -> Final state : ");
            utils::print_state(state_grids[worker_id], profile_size * BL_RANK,
                               BL_RANK);
            printf("   -> Rhs Jacobian : ");
            utils::print_matrix_row_major(
                solver_resources[worker_id].matrix.GetData(), BL_RANK, BL_RANK,
                2);
          }
        }
        return SearchOutcome{true, worker_id, best_profile_size, best_guess};
      }

    } else {
      if (verbose)
        printf("Iter #%d, line search failed . Aborting.\n", iter + 1);
      break;
    }

    iter++;
  }

  bool success = (snorm < rtol);
  if (verbose)
    printf("  -> %s search.\n", (success) ? "Successful" : "Unsuccesful");

  // If the search did not converge, make sure the profile in
  // state_grids[worker_id] is that of the best profile found.
  if (!success) {
    profile_params.SetInitialValues(best_guess);
    int profile_size = DevelopProfile(profile_params, score, worker_id);
    devel_count++;
  }

  printf("\n\n# So far: %d calls to devel, %d calls with usable gradient #\n\n",
         devel_count, devel_with_grad_count);

  return SearchOutcome{success, worker_id, best_profile_size, best_guess};
}

//////
// 2D compute functions
//

vector<SearchOutcome> BoundaryLayer::Compute(
    const BoundaryData &boundary_data, ProfileParams &profile_params,
    SearchParams &search_params, vector<vector<double>> &bl_state_grid) {
  SolveType solve_type = profile_params.solve_type;

  if (solve_type == SolveType::LocallySimilar) {
    return ComputeLocalSimilarity(boundary_data, profile_params, search_params,
                                  bl_state_grid);
  }

  if (solve_type == SolveType::DifferenceDifferential) {
    return ComputeDifferenceDifferential(boundary_data, profile_params,
                                         search_params, bl_state_grid);
  }

  printf("\nCompute: SolveType input not recognized.\n");
  return {};
}

template <std::size_t MAX_NB_WORKERS>
void compute_args_are_consistent(
    const BoundaryData &boundary_data,
    const vector<vector<double>> &bl_state_grid,
    const array<vector<double>, MAX_NB_WORKERS> &eta_grids,
    const array<vector<double>, MAX_NB_WORKERS> &state_grids) {
  // Output arrays should have consistent dimensions
  assert(bl_state_grid.size() >= 1);
  int eta_dim = eta_grids[0].size();

  assert(BL_RANK * eta_dim == state_grids[0].size());
  assert(bl_state_grid[0].size() == state_grids[0].size());

  // Checking edge_field has the correct dimensions
  const int xi_dim = boundary_data.xi_dim;
  assert(xi_dim == bl_state_grid.size());
}

void print_summary(const vector<SearchOutcome> &search_outcomes) {
  if (std::any_of(
          search_outcomes.begin(), search_outcomes.end(),
          [](const SearchOutcome &outcome) { return !outcome.success; })) {
    printf("Could not solve stations ");
    int station_id = 0;
    for (const SearchOutcome &outcome : search_outcomes) {
      if (!outcome.success)
        printf("%d, ", station_id);
      station_id++;
    }
    printf("\n");
  }
}

/*
 * This function computes the whole 2D profile using the local-similarity method
 *
 * The output should be grid data (xi, eta) -> (x, y, state)
 *
 * Function parameters should be:
 *  1. grid (xi, eta)
 *  2. wall properties wrt xi
 *  3. edge properties wrt xi
 */
vector<SearchOutcome> BoundaryLayer::ComputeLocalSimilarity(
    const BoundaryData &boundary_data, ProfileParams &profile_params,
    SearchParams &search_params, vector<vector<double>> &bl_state_grid) {
  // Consistency check
  compute_args_are_consistent(boundary_data, bl_state_grid, eta_grids,
                              state_grids);

  // Data structures
  const int xi_dim = boundary_data.xi_dim;

  vector<SearchOutcome> search_outcomes(xi_dim);

  const bool verbose = search_params.verbose;

  // Worker function
  auto lsim_task = [&profile_params, &search_params, &boundary_data,
                    &bl_state_grid, verbose,
                    this](int xi_id, array<double, 2> guess = {{0.5, 0.5}}) {
    if (verbose)
      printf("###\n# station #%d - start\n#\n\n", xi_id);

    // Set local profile settings
    profile_params.ReadEdgeConditions(boundary_data.edge_field,
                                      xi_id * EDGE_FIELD_RANK);
    profile_params.ReadWallConditions(boundary_data.wall_field, xi_id);

    if (!profile_params.AreValid()) {
      printf("Invalid edge conditions at station #%d. Abort\n", xi_id);
      return SearchOutcome{false, 0, 0};
    }

    if (verbose)
      profile_params.PrintODEFactors();

    // Call search method
    SearchOutcome outcome = ProfileSearch(profile_params, search_params, guess);

    if (verbose) {
      if (outcome.success) {
        printf("\n\n# station #%d: SUCCESS.\n###\n\n", xi_id);
      } else {
        printf("\n\n# station #%d: FAIL.\n###\n\n", xi_id);
      }
    }

    // Copy to output
    std::copy(state_grids[outcome.worker_id].begin(),
              state_grids[outcome.worker_id].end(),
              bl_state_grid[xi_id].begin());

    return std::move(outcome);
  };

  printf("##################################\n"
         "# Local-Similarity Solve (START) #\n\n");

  // First station -> Self-Similar solve
  profile_params.solve_type = SolveType::SelfSimilar;

  SearchOutcome outcome = lsim_task(0);

  search_outcomes[0] = std::move(outcome);

  // Following stations -> Locally-Similar solves
  profile_params.solve_type = SolveType::LocallySimilar;

  printf("# FORWARD LOOP #\n\n");

  for (int xi_id = 1; xi_id < xi_dim; xi_id++) {

    outcome = lsim_task(xi_id, outcome.guess);

    // Assign outcome
    search_outcomes[xi_id] = std::move(outcome);
  }

  print_summary(search_outcomes);

  printf("\n# INVERSE LOOP #\n\n");

  for (int xi_id = xi_dim - 2; xi_id > 0; xi_id--) {
    if (search_outcomes[xi_id].success)
      continue;

    outcome = lsim_task(xi_id, search_outcomes[xi_id + 1].guess);

    if (outcome.success) {
      printf("# Station #%d was solved from station %d!\n", xi_id, xi_id + 1);
      search_outcomes[xi_id] = outcome;
    } else {
      printf("# Station #%d could not be solved from station %d!\n", xi_id,
             xi_id + 1);
    }
  }

  printf("\n");
  print_summary(search_outcomes);

  printf("\n# Local-Similarity Solve (END) #\n"
         "################################\n\n");

  return std::move(search_outcomes);
}

/*
 * This function computes the whole 2D profile using a parallel
 * implementation of the local-similarity method with gradient
 * search.
 */
vector<SearchOutcome> BoundaryLayer::ComputeLocalSimilarityParallel(
    const BoundaryData &boundary_data, ProfileParams &profile_params,
    SearchParams &search_params, vector<vector<double>> &bl_state_grid,
    const int &nb_workers) {
  compute_args_are_consistent(boundary_data, bl_state_grid, eta_grids,
                              state_grids);

  bool verbose = search_params.verbose;

  // Data structures
  const int xi_dim = boundary_data.xi_dim;

  assert(nb_workers <= MAX_NB_WORKERS);

  vector<array<double, 2>> guess_buffers(nb_workers, {{0.5, 0.5}});

  vector<SearchOutcome> search_outcomes(xi_dim);

  // Worker function
  auto lsim_loop_task = [&search_params, &boundary_data, &bl_state_grid,
                         &guess_buffers, &search_outcomes, nb_workers, verbose,
                         this](int xi0_id, int xi_end, int worker_id,
                               ProfileParams profile_params) {
    array<double, 2> &best_guess = guess_buffers[worker_id];

    for (int xi_id = xi0_id; xi_id < xi_end; xi_id += nb_workers) {

      if (verbose) {
        printf("###\n# station #%d - start\n#\n\n", xi_id);
        profile_params.PrintODEFactors();
      }

      // Set local profile settings
      profile_params.ReadEdgeConditions(boundary_data.edge_field,
                                        xi_id * EDGE_FIELD_RANK);
      profile_params.ReadWallConditions(boundary_data.wall_field, xi_id);

      if (!profile_params.AreValid()) {
        printf("Invalid edge conditions at station #%d. \n", xi_id);
        return;
      }

      // Call search method
      SearchOutcome outcome = GradientProfileSearch(
          profile_params, search_params, best_guess, worker_id);

      if (verbose) {
        if (outcome.success) {
          printf("\n\n# station #%d: SUCCESS.\n###\n\n", xi_id);
        } else {
          printf("\n\n# station #%d: FAIL.\n###\n\n", xi_id);
        }
      }

      // Copy to output
      std::copy(state_grids[outcome.worker_id].begin(),
                state_grids[outcome.worker_id].end(),
                bl_state_grid[xi_id].begin());

      search_outcomes[xi_id] = std::move(outcome);
    }
  };

  if (verbose) {
    printf("##################################\n"
           "# Local-Similarity Solve (START) #\n\n");
  }

  // First station -> Self-Similar solve
  profile_params.solve_type = SolveType::SelfSimilar;

  lsim_loop_task(0, 1, 0, profile_params);

  // Share the solution as initial guess to the other workers
  for (int worker_id = 1; worker_id < nb_workers; worker_id++) {
    std::copy(guess_buffers[0].begin(), guess_buffers[0].end(),
              guess_buffers[worker_id].begin());
  }

  // Following stations -> Locally-Similar solves
  profile_params.solve_type = SolveType::LocallySimilar;

  // Get the threads working
  vector<std::future<void>> futures;
  for (int worker_id = 0; worker_id < nb_workers; worker_id++) {
    futures.emplace_back(std::async(std::launch::async, lsim_loop_task,
                                    1 + worker_id, xi_dim, worker_id,
                                    profile_params));
  }

  // Wait for the threads
  std::for_each(futures.begin(), futures.end(), [](auto &ftr) { ftr.wait(); });

  if (verbose) {
    printf("\n# Local-Similarity Solve (END) #\n"
           "################################\n\n");
  }

  print_summary(search_outcomes);

  return std::move(search_outcomes);
}

/*
 * This function computes the whole 2D profile using the
 * difference-differential method.
 *
 * The output should be grid data (xi, eta) -> (x, y, state)
 *
 * Function parameters should be:
 *  1. grid (xi, eta)
 *  2. wall properties wrt xi
 *  3. edge properties wrt xi
 */
inline void ComputeDiffField_BE(const int &xi_id, const int &nb_steps,
                                const TimeScheme &time_scheme,
                                const vector<double> &edge_field,
                                const vector<vector<double>> &bl_state_grid,
                                vector<double> &field_grid) {
  assert(xi_id > 0);

  int field_offset = 0;
  int state_offset = 0;

  const int xi_offset = xi_id * EDGE_FIELD_RANK + EDGE_XI_ID;
  const double xi_val = edge_field[xi_offset];
  const double xim1_val = edge_field[xi_offset - EDGE_FIELD_RANK];

  double theta = SCHEME_THETA[static_cast<int>(time_scheme)];

  const vector<double> &state_grid_m1 = bl_state_grid[xi_id - 1];

  // 1st order backward difference
  const double be_factor = 2. * xi_val / (xi_val - xim1_val);

  for (int step_id = 0; step_id < nb_steps; step_id++) {

    double fp_im1 = (1. - theta) * state_grid_m1[state_offset + FP_ID] +
                    theta * state_grid_m1[state_offset + BL_RANK + FP_ID];

    double f_im1 = (1. - theta) * state_grid_m1[state_offset + F_ID] +
                   theta * state_grid_m1[state_offset + BL_RANK + F_ID];

    double g_im1 = (1. - theta) * state_grid_m1[state_offset + G_ID] +
                   theta * state_grid_m1[state_offset + BL_RANK + G_ID];

    field_grid[field_offset + FIELD_M0_ID] = be_factor;
    field_grid[field_offset + FIELD_M1_ID] = -be_factor * fp_im1;
    field_grid[field_offset + FIELD_S0_ID] = be_factor;
    field_grid[field_offset + FIELD_S1_ID] = -be_factor * f_im1;

    field_grid[field_offset + FIELD_E0_ID] = be_factor;
    field_grid[field_offset + FIELD_E1_ID] = -be_factor * g_im1;

    field_offset += FIELD_RANK;
    state_offset += BL_RANK;
  }
}

inline void ComputeDiffField_LG2(const int &xi_id, const int &nb_steps,
                                 const TimeScheme &time_scheme,
                                 const vector<double> &edge_field,
                                 const vector<vector<double>> &bl_state_grid,
                                 vector<double> &field_grid) {
  assert(xi_id > 0);

  int field_offset = 0;
  int state_offset = 0;

  const int xi_offset = xi_id * EDGE_FIELD_RANK + EDGE_XI_ID;
  const double xi_val = edge_field[xi_offset];
  const double xim1_val = edge_field[xi_offset - EDGE_FIELD_RANK];

  double theta = SCHEME_THETA[static_cast<int>(time_scheme)];

  if (xi_id == 1) {
    ComputeDiffField_BE(xi_id, nb_steps, time_scheme, edge_field, bl_state_grid,
                        field_grid);
    return;
  }

  const double xim2_val = edge_field[xi_offset - 2 * EDGE_FIELD_RANK];

  assert(xi_val > xim1_val && xim1_val > xim2_val);

  // 2nd order backward difference
  const double lag0 = 2. * xi_val * (2. * xi_val - xim1_val - xim2_val) /
                      ((xi_val - xim1_val) * (xi_val - xim2_val));
  const double lag1 = 2. * xi_val * (xim2_val - xi_val) /
                      ((xi_val - xim1_val) * (xim1_val - xim2_val));
  const double lag2 = 2. * xi_val * (xi_val - xim1_val) /
                      ((xi_val - xim2_val) * (xim1_val - xim2_val));

  const vector<double> &state_grid_m1 = bl_state_grid[xi_id - 1];
  const vector<double> &state_grid_m2 = bl_state_grid[xi_id - 2];

  for (int step_id = 0; step_id < nb_steps; step_id++) {

    double fp_im1 = (1 - theta) * state_grid_m1[state_offset + FP_ID] +
                    theta * state_grid_m1[state_offset + BL_RANK + FP_ID];
    double f_im1 = (1 - theta) * state_grid_m1[state_offset + F_ID] +
                   theta * state_grid_m1[state_offset + BL_RANK + F_ID];
    double g_im1 = (1 - theta) * state_grid_m1[state_offset + G_ID] +
                   theta * state_grid_m1[state_offset + BL_RANK + G_ID];

    double fp_im2 = (1 - theta) * state_grid_m2[state_offset + FP_ID] +
                    theta * state_grid_m2[state_offset + BL_RANK + FP_ID];
    double f_im2 = (1 - theta) * state_grid_m2[state_offset + F_ID] +
                   theta * state_grid_m2[state_offset + BL_RANK + F_ID];
    double g_im2 = (1 - theta) * state_grid_m2[state_offset + G_ID] +
                   theta * state_grid_m2[state_offset + BL_RANK + G_ID];

    field_grid[field_offset + FIELD_M0_ID] = lag0;
    field_grid[field_offset + FIELD_M1_ID] = lag1 * fp_im1 + lag2 * fp_im2;
    field_grid[field_offset + FIELD_S0_ID] = lag0;
    field_grid[field_offset + FIELD_S1_ID] = lag1 * f_im1 + lag1 * f_im2;

    field_grid[field_offset + FIELD_E0_ID] = lag0;
    field_grid[field_offset + FIELD_E1_ID] = lag1 * g_im1 + lag2 * g_im2;

    field_offset += FIELD_RANK;
    state_offset += BL_RANK;
  }
}

vector<SearchOutcome> BoundaryLayer::ComputeDifferenceDifferential(
    const BoundaryData &boundary_data, ProfileParams &profile_params,
    SearchParams &search_params, vector<vector<double>> &bl_state_grid) {

  bool verbose = search_params.verbose;

  compute_args_are_consistent(boundary_data, bl_state_grid, eta_grids,
                              state_grids);

  std::fill(field_grid.begin(), field_grid.end(), 0.);

  const int xi_dim = boundary_data.xi_dim;

  vector<SearchOutcome> search_outcomes(xi_dim);

  // Function to compute Diff-Diff model coefficients
  using FieldFunction = void (*)(
      const int &, const int &, const TimeScheme &, const vector<double> &,
      const vector<vector<double>> &, vector<double> &);

  FieldFunction compute_diff_field = ComputeDiffField_BE;

  // Worker function
  double scale = 1.0;
  auto diff_diff_task = [&boundary_data, &profile_params, &search_params,
                         &bl_state_grid, verbose, &scale,
                         this](const int &xi_id,
                               array<double, 2> best_guess = {{0.5, 0.5}}) {
    if (verbose) {
      printf("###\n# station #%d - start\n#\n\n", xi_id);
    }

    // Set local profile settings
    profile_params.ReadEdgeConditions(boundary_data.edge_field,
                                      xi_id * EDGE_FIELD_RANK);
    profile_params.ReadWallConditions(boundary_data.wall_field, xi_id);

    if (!profile_params.AreValid()) {
      printf("Invalid edge conditions. Abort\n");
      return SearchOutcome{false, 0, 1, best_guess};
    }

    // Call search method
    SearchOutcome outcome =
        ProfileSearch(profile_params, search_params, best_guess);

    // If the search converged to a 'trimmed' profile, finish it manually
    if (outcome.success && outcome.profile_size < _max_nb_steps &&
        scale == 1.0) {
      printf("Profile converged but terminated early!\n");

      vector<double> &state_grid = state_grids[outcome.worker_id];

      const int eta_start = outcome.profile_size;
      const double f_start = state_grid[eta_start * BL_RANK + F_ID];
      const double fp_ref = state_grid[eta_start * BL_RANK + FP_ID];
      const double g_ref = state_grid[eta_start * BL_RANK + G_ID];

      for (int eta_id = eta_start; eta_id < _max_nb_steps + 1; eta_id++) {
        state_grid[eta_id * BL_RANK + F_ID] =
            f_start + (eta_id - eta_start) * profile_params.max_step * fp_ref;
        state_grid[eta_id * BL_RANK + G_ID] = g_ref;
        state_grid[eta_id * BL_RANK + FP_ID] = fp_ref;
      }

      outcome.profile_size = _max_nb_steps;
    }

    // Copy profile to output vector
    std::copy(state_grids[outcome.worker_id].begin(),
              state_grids[outcome.worker_id].end(),
              bl_state_grid[xi_id].begin());

    if (verbose) {
      if (!outcome.success) {
        printf("\n#\n# station #%d: Unsuccessful search.\n###\n\n", xi_id);
      } else {
        printf("\n#\n# station #%d: Successful search.\n###\n\n", xi_id);
      }
    }

    return std::move(outcome);
  };

  printf("#########################################\n"
         "# Difference-Differential Solve (START) #\n\n");

  // First solve is self-similar
  profile_params.solve_type = SolveType::SelfSimilar;

  search_outcomes[0] = diff_diff_task(0);

  if (!search_outcomes[0].success)
    return std::move(search_outcomes);

  // Remaining solves
  profile_params.solve_type = SolveType::DifferenceDifferential;

  for (int xi_id = 1; xi_id < xi_dim; xi_id++) {

    // Careful to not develop the profile farther than at the previous station
    // since the xi derivative terms will not be defined.
    if (profile_params.nb_steps != search_outcomes[xi_id - 1].profile_size) {
      profile_params.nb_steps = search_outcomes[xi_id - 1].profile_size;
      printf(" -> Profile size lowered to %d.\n", profile_params.nb_steps);
    }

    compute_diff_field(xi_id, profile_params.nb_steps, profile_params.scheme,
                       boundary_data.edge_field, bl_state_grid, field_grid);

    // Call search method
    search_outcomes[xi_id] = diff_diff_task(xi_id);

    if (search_outcomes[xi_id].success) {
      printf("SUCCESS at station #%d.\n", xi_id);
      continue;
    }

    printf("FAIL at station #%d. Scaling the source term...\n", xi_id);

    // At this point the search wasn't successfull
    int nb_attemps = 0;
    scale = 1.;
    SearchOutcome temp_outcome;

    // Make the problem easier
    const int max_nb_attemps = 10;
    while (nb_attemps < max_nb_attemps) {
      scale *= 0.5;

      std::transform(field_grid.begin(), field_grid.end(), field_grid.begin(),
                     [scale](double val) { return val * scale; });

      temp_outcome = diff_diff_task(xi_id);
      if (temp_outcome.success == true) {
        printf(" -> Success with a factor %.2e.\n", scale);
        break;
      }

      nb_attemps++;
    }

    // If you found a solvable state
    if (nb_attemps < max_nb_attemps) {

      // Work your way back up!
      scale *= 2;
      while (scale <= 1) {
        std::transform(field_grid.begin(), field_grid.end(), field_grid.begin(),
                       [scale](double val) { return val * scale; });

        temp_outcome = diff_diff_task(xi_id, temp_outcome.guess);

        if (!temp_outcome.success) {
          printf("Couldn't solve back for factor %.2e.\n", scale);
          break;
        } else {
          printf("Solved for factor %.2e.\n", scale);
        }

        scale *= 2;
      }
    } else {
      printf("Couldn't solve a simpler system.\n");
    }

    if (scale > 1) {
      printf("Well done! System solved! f''(0) = %.3e, g'(0) = %.3e\n\n",
             temp_outcome.guess[0], temp_outcome.guess[1]);
      search_outcomes[xi_id] = temp_outcome;
    }

    if (!search_outcomes[xi_id].success) {
      // Write field coeffs to file

      static vector<LabelIndex> field_labels = {
          {"m0", FIELD_M0_ID}, {"m1", FIELD_M1_ID}, {"s0", FIELD_S0_ID},
          {"s1", FIELD_S1_ID}, {"e0", FIELD_E0_ID}, {"e1", FIELD_E1_ID},
      };

      WriteH5("field.h5", field_grid, field_labels, _max_nb_steps, FIELD_RANK,
              "fields");

      break;
    }
  }

  printf("\n# Difference-Differential Solve (END) #\n"
         "#######################################\n\n");

  return std::move(search_outcomes);
}