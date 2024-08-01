#include "flat_plate.h"
#include "newton_solver.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <tuple>

FlatPlate::FlatPlate(int max_nb_steps)
    : _max_nb_steps(max_nb_steps), compute_rhs(compute_rhs_default),
      initialize(initialize_default),
      compute_rhs_jacobian(compute_rhs_jacobian_default),
      state_grids(_max_nb_workers,
                  std::vector<double>(FLAT_PLATE_RANK * (1 + _max_nb_steps))),
      eta_grids(_max_nb_workers, std::vector<double>(1 + _max_nb_steps)),
      rhs_vecs(_max_nb_workers, std::vector<double>(FLAT_PLATE_RANK)) {}

FlatPlate::FlatPlate(int max_nb_steps, RhsFunction compute_rhs_fun,
                     InitializeFunction init_fun,
                     RhsJacobianFunction jacobian_fun)
    : _max_nb_steps(max_nb_steps), compute_rhs(compute_rhs_fun),
      compute_rhs_jacobian(jacobian_fun), initialize(init_fun),
      state_grids(_max_nb_workers,
                  std::vector<double>(FLAT_PLATE_RANK * (1 + _max_nb_steps))),
      eta_grids(_max_nb_workers, std::vector<double>(1 + _max_nb_steps)),
      rhs_vecs(_max_nb_workers, std::vector<double>(FLAT_PLATE_RANK)) {}

void FlatPlate::InitializeState(ProfileParams &profile_params, int worker_id) {
  initialize(profile_params, state_grids[worker_id]);
}

bool FlatPlate::DevelopProfile(ProfileParams &profile_params,
                               std::vector<double> &score, int worker_id) {
  if (profile_params.scheme == TimeScheme::Explicit) {
    return DevelopProfileExplicit(profile_params, score, worker_id);
  } else if (profile_params.scheme == TimeScheme::Implicit) {
    return DevelopProfileImplicit(profile_params, score, worker_id);
  } else {
    printf("Time Scheme not recognized!");
    return false;
  }
}

bool FlatPlate::DevelopProfileExplicit(ProfileParams &profile_params,
                                       std::vector<double> &score,
                                       int worker_id) {
  assert(profile_params.AreValid());

  vector<double> &state_grid = state_grids[worker_id];
  vector<double> &eta_grid = eta_grids[worker_id];
  vector<double> &rhs = rhs_vecs[worker_id];

  int nb_steps = eta_grid.size() - 1;

  assert(state_grid.size() / (nb_steps + 1) == FLAT_PLATE_RANK);

  InitializeState(profile_params, worker_id);
  double max_step = profile_params.max_step;

  int step_id = 0;
  int offset = 0;
  while (step_id < nb_steps) {

    // Compute rhs and limit time step
    double eta_step = std::min(
        max_step, compute_rhs(state_grid, rhs, offset, profile_params));

    // Evolve state/grid forward
    eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;
    for (int var_id = 0; var_id < FLAT_PLATE_RANK; ++var_id) {
      state_grid[offset + FLAT_PLATE_RANK + var_id] =
          state_grid[offset + var_id] + eta_step * rhs[var_id];
    }

    // Update indexing
    offset += FLAT_PLATE_RANK;
    step_id += 1;
  }

  // Check convergence
  double rate = sqrt(rhs[FP_ID] * rhs[FP_ID] + rhs[G_ID] * rhs[G_ID]);
  bool converged = rate < 1e-3;

  // Compute score
  score[0] = state_grid[offset + FP_ID] - 1.;
  score[1] = state_grid[offset + G_ID] - 1;

  return converged;
}

bool FlatPlate::DevelopProfileImplicit(ProfileParams &profile_params,
                                       std::vector<double> &score,
                                       int worker_id) {
  assert(profile_params.AreValid());

  vector<double> &state_grid = state_grids[worker_id];
  vector<double> &eta_grid = eta_grids[worker_id];
  vector<double> &rhs = rhs_vecs[worker_id];

  int nb_steps = eta_grid.size() - 1;

  assert(state_grid.size() / (nb_steps + 1) == FLAT_PLATE_RANK);

  InitializeState(profile_params, worker_id);
  double eta_step = profile_params.max_step;

  int step_id = 0;
  int offset = 0;

  // Setup nonlinear solver functions
  int xdim = FLAT_PLATE_RANK;

  auto objective_fun = [xdim, worker_id, &eta_step, &offset, &profile_params,
                        &state_grid, this](const std::vector<double> &state,
                                           std::vector<double> &residual) {
    // U^{n+1} - U^{n} = R(U^{n+1})
    compute_rhs(state, residual, 0, profile_params);
    for (int idx = 0; idx < xdim; idx++) {
      residual[idx] *= -eta_step;
      residual[idx] += state[idx] - state_grid[offset + idx];
    };
  };

  auto limit_update_fun = [xdim](const std::vector<double> &state,
                                 const std::vector<double> &state_varn) {
    double alpha = 1.;

    if (state_varn[FP_ID] < 0) {
      alpha = std::min(alpha, 0.2 * state[FP_ID] / (-state_varn[FP_ID]));
    }
    alpha = std::min(alpha, 0.2 * state[G_ID] / fabs(state_varn[G_ID] + 1e-30));

    return alpha;
  };

  auto jacobian_fun = [xdim, &eta_step, &offset, &profile_params,
                       this](const std::vector<double> &state,
                             DenseMatrix &matrix) {
    std::vector<double> &matrix_data = matrix.GetData();

    compute_rhs_jacobian(state, matrix_data, profile_params);
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

  std::vector<double> solution_buffer(xdim, 0.);
  for (int idx = 0; idx < xdim; idx++) {
    solution_buffer[idx] = state_grid[idx];
  }

  NewtonParams newton_params;

  // Time loop
  while (step_id < nb_steps) {

    // Solve nonlinear system
    bool pass = NewtonSolveDirect(solution_buffer, objective_fun, jacobian_fun,
                                  limit_update_fun, newton_params);
    if (!pass) {
      score[0] = state_grid[offset + FP_ID] - 1.0;
      score[1] = state_grid[offset + G_ID] - 1.0;

      return false;
    }

    // Evolve state/grid forward
    eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;

    for (int var_id = 0; var_id < FLAT_PLATE_RANK; ++var_id) {
      state_grid[offset + FLAT_PLATE_RANK + var_id] = solution_buffer[var_id];
    }

    // Update indexing
    offset += FLAT_PLATE_RANK;
    step_id += 1;
  }

  compute_rhs(solution_buffer, rhs, 0, profile_params);
  double rate = sqrt(rhs[FP_ID] * rhs[FP_ID] + rhs[G_ID] * rhs[G_ID]);
  bool converged = rate < 1e-3;

  // Compute score
  score[0] = state_grid[offset + FP_ID] - 1.;
  score[1] = state_grid[offset + G_ID] - 1;

  return converged;
}

int FlatPlate::BoxProfileSearch(ProfileParams &profile_params,
                                SearchWindow &window,
                                SearchParams &search_params,
                                std::vector<double> &best_guess) {
  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  // Fetch search window parameters
  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  // Temporary arrays
  std::vector<double> initial_guess(2, 0.0);
  std::vector<double> score(2, 0.0);

  double res_norm;

  for (int iter = 0; iter < max_iter; iter++) {

    double delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    double delta_gp = (gp_max - gp_min) / (ydim - 1);

    double fpp0 = fpp_min;
    double gp0 = gp_min;

    double min_res_norm = 1e3;

    int min_fid = 0;
    int min_gid = 0;

    // (1 / 2) Develop profiles on square grid of initial conditions
    for (int fid = 0; fid < xdim; fid++) {
      initial_guess[0] = fpp0;

      gp0 = gp_min;
      for (int gid = 0; gid < ydim; gid++) {
        initial_guess[1] = gp0;

        profile_params.SetInitialValues(initial_guess);
        InitializeState(profile_params);

        bool converged = DevelopProfile(profile_params, score);

        if (converged) {
          res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

          if (res_norm < min_res_norm) {

            min_res_norm = res_norm;
            min_fid = fid;
            min_gid = gid;

            if (res_norm < rtol) {
              printf("Solution found: f''(0)=%.6f, g'(0)=%.6f.\n", fpp0, gp0);
              best_guess[0] = fpp0;
              best_guess[1] = gp0;
              return 0;
            }
          }
        }

        gp0 += delta_gp;
      }

      fpp0 += delta_fpp;
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

    printf("Iter %d - score=%.2e, f''(0)=%.3f, g'(0)=%.3f, next "
           "window "
           "=[[%.3f, %.3f], [%.3f, %.3f]\n",
           iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
  }

  return 0;
}

#include <future>
#include <thread>

int FlatPlate::BoxProfileSearchParallel(ProfileParams &profile_params,
                                        SearchWindow &window,
                                        SearchParams &search_params,
                                        std::vector<double> &best_guess) {
  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  // Fetch search window parameters
  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  int best_worker_id = -1;

  for (int iter = 0; iter < max_iter; iter++) {

    double delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    double delta_gp = (gp_max - gp_min) / (ydim - 1);

    // (1 / 2) Develop profiles on square grid of initial conditions
    auto local_search_task = [fpp_min, gp_min, delta_fpp, delta_gp, rtol,
                              profile_params,
                              this](const int fid_start, const int fid_end,
                                    const int gid_start, const int gid_end,
                                    const int worker_id) mutable {
      std::vector<double> initial_guess(2, 0.0);
      std::vector<double> score(2, 0.0);

      double res_norm;
      double min_res_norm = 1e30;

      int min_fid = fid_start, min_gid = gid_start;

      double fpp0 = fpp_min + fid_start * delta_fpp;
      for (int fid = fid_start; fid < fid_end; fid++) {
        initial_guess[0] = fpp0;

        double gp0 = gp_min + gid_start * delta_gp;
        for (int gid = gid_start; gid < gid_end; gid++) {
          initial_guess[1] = gp0;

          profile_params.SetInitialValues(initial_guess);
          InitializeState(profile_params, worker_id);

          bool converged = DevelopProfile(profile_params, score, worker_id);

          if (converged) {
            res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

            if (res_norm < min_res_norm) {

              min_res_norm = res_norm;
              min_fid = fid;
              min_gid = gid;

              if (res_norm < rtol) {
                return SearchResult(min_res_norm, min_fid, min_gid, worker_id);
              }
            }
          }

          gp0 += delta_gp;
        }

        fpp0 += delta_fpp;
      }
      return SearchResult(min_res_norm, min_fid, min_gid, worker_id);
    };

    double min_res_norm = 1e30;
    int min_fid = 0, min_gid = 0;

    // 4 threads
    std::vector<std::future<SearchResult>> futures;

    futures.emplace_back(std::async(std::launch::async, local_search_task, 0,
                                    xdim / 2, 0, ydim / 2, 0));
    futures.emplace_back(std::async(std::launch::async, local_search_task,
                                    xdim / 2, xdim, 0, ydim / 2, 1));
    futures.emplace_back(std::async(std::launch::async, local_search_task, 0,
                                    xdim / 2, ydim / 2, ydim, 2));
    futures.emplace_back(std::async(std::launch::async, local_search_task,
                                    xdim / 2, xdim, ydim / 2, ydim, 3));

    for (auto &ftr : futures) {
      SearchResult result = ftr.get();
      double res_norm = result.res_norm;
      if (res_norm < min_res_norm) {
        min_res_norm = res_norm;
        min_fid = result.xid;
        min_gid = result.yid;
        best_worker_id = result.worker_id;
      }
    }

    // If your best guess is good enough, terminate.
    if (min_res_norm < rtol) {
      best_guess[0] = fpp_min + min_fid * delta_fpp;
      best_guess[1] = gp_min + min_gid * delta_gp;
      printf("Solution found: f''(0)=%.6f, g'(0)=%.6f.\n", best_guess[0],
             best_guess[1]);

      return best_worker_id;
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

    printf("Iter %d - score=%.2e, f''(0)=%.3f, g'(0)=%.3f, next "
           "window "
           "=[[%.3f, %.3f], [%.3f, %.3f]\n",
           iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
  }

  return best_worker_id;
}

#include "message_queue.h"
#include <memory>

int FlatPlate::BoxProfileSearchParallelWithQueues(
    ProfileParams &profile_params, SearchWindow &window,
    SearchParams &search_params, std::vector<double> &best_guess) {
  // Fetch search parameters
  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  // Fetch search window parameters
  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  int best_worker_id = -1;

  // Define queues for SearchInput and SearchResult
  std::shared_ptr<MessageQueue<SearchInput>> work_queue(
      new MessageQueue<SearchInput>());
  std::shared_ptr<MessageQueue<SearchResult>> result_queue(
      new MessageQueue<SearchResult>());

  // Spawn worker threads
  auto local_search_task = [rtol, profile_params, work_queue, result_queue,
                            this](const int worker_id) mutable {
    while (true) {
      SearchInput inputs = work_queue->fetch();

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

      std::vector<double> initial_guess(2, 0.0);
      std::vector<double> score(2, 0.0);

      double res_norm;
      double min_res_norm = 1e30;

      int min_fid = fid_start, min_gid = gid_start;

      double fpp0 = fpp_min + fid_start * delta_fpp;
      for (int fid = fid_start; fid < fid_end; fid++) {
        initial_guess[0] = fpp0;

        double gp0 = gp_min + gid_start * delta_gp;
        for (int gid = gid_start; gid < gid_end; gid++) {
          initial_guess[1] = gp0;

          profile_params.SetInitialValues(initial_guess);
          InitializeState(profile_params, worker_id);

          bool converged = DevelopProfile(profile_params, score, worker_id);

          if (converged) {
            res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

            if (res_norm < min_res_norm) {

              min_res_norm = res_norm;
              min_fid = fid;
              min_gid = gid;

              if (res_norm < rtol) {
                result_queue->send(
                    SearchResult(min_res_norm, min_fid, min_gid, worker_id));

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
        result_queue->send(
            SearchResult(min_res_norm, min_fid, min_gid, worker_id));
      }
    }
  };

  std::vector<std::future<void>> futures;
  for (int worker_id = 0; worker_id < 4; worker_id++) {
    futures.emplace_back(
        std::async(std::launch::async, local_search_task, worker_id));
  }

  for (int iter = 0; iter < max_iter; iter++) {

    // Compute step sizes
    double delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    double delta_gp = (gp_max - gp_min) / (ydim - 1);

    // Send work orders
    work_queue->send(SearchInput(fpp_min, delta_fpp, gp_min, delta_gp, 0,
                                 xdim / 2, 0, ydim / 2));
    work_queue->send(SearchInput(fpp_min, delta_fpp, gp_min, delta_gp, xdim / 2,
                                 xdim, 0, ydim / 2));
    work_queue->send(SearchInput(fpp_min, delta_fpp, gp_min, delta_gp, 0,
                                 xdim / 2, ydim / 2, ydim));
    work_queue->send(SearchInput(fpp_min, delta_fpp, gp_min, delta_gp, xdim / 2,
                                 xdim, ydim / 2, ydim));

    // Fetch best guesses from workers
    double min_res_norm = 1e30;
    int min_fid = 0, min_gid = 0;

    int processed_results = 0;
    while (processed_results < 4) {
      SearchResult result = result_queue->fetch();

      if (result.res_norm < min_res_norm) {
        min_res_norm = result.res_norm;
        min_fid = result.xid;
        min_gid = result.yid;
        best_worker_id = result.worker_id;
      }

      processed_results += 1;
    }

    // If your best guess is good enough, terminate.
    if (min_res_norm < rtol) {
      best_guess[0] = fpp_min + min_fid * delta_fpp;
      best_guess[1] = gp_min + min_gid * delta_gp;
      printf("Solution found: f''(0)=%.6f, g'(0)=%.6f.\n", best_guess[0],
             best_guess[1]);
      work_queue->stop();
      return best_worker_id;
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

    printf("Iter %d - score=%.2e, f''(0)=%.3f, g'(0)=%.3f, next "
           "window "
           "=[[%.3f, %.3f], [%.3f, %.3f]\n",
           iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
  }

  work_queue->stop();

  std::for_each(futures.begin(), futures.end(),
                [](std::future<void> &ftr) { ftr.wait(); });

  return best_worker_id;
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

void FlatPlate::Compute(std::vector<double> &xi_grid,
                        std::vector<double> &edge_field,
                        SearchParams &search_params,
                        std::vector<std::vector<double>> &bl_state_grid) {
  // Output arrays should have consistent dimensions
  assert(bl_state_grid.size() >= 1);
  int eta_dim = eta_grids[0].size();

  assert(FLAT_PLATE_RANK * eta_dim == state_grids[0].size());
  assert(bl_state_grid[0].size() == state_grids[0].size());

  // Checking edge_field has the correct dimensions
  int xi_dim = xi_grid.size();
  assert(xi_dim == bl_state_grid.size());
  assert(edge_field.size() == xi_dim * 6);

  // Parameters
  ProfileParams profile_params;
  profile_params.SetDefault();

  SearchWindow search_window;
  search_window.SetDefault();

  //
  std::vector<double> best_guess(2, 0.5);

  int offset = 0;
  for (int xi_id = 0; xi_id < xi_dim; xi_id++) {
    // Set local profile settings
    profile_params.ue = edge_field[offset + 0];
    profile_params.he = edge_field[offset + 1];
    profile_params.pe = edge_field[offset + 2];
    profile_params.xi = edge_field[offset + 3];
    profile_params.due_dxi = edge_field[offset + 4];
    profile_params.dhe_dxi = edge_field[offset + 5];

    // Call search method
    int worker_id = BoxProfileSearchParallel(profile_params, search_window,
                                             search_params, best_guess);

    // Copy profile to output vector
    bl_state_grid[xi_id] = state_grids[worker_id];

    // Define search window for next iterations
    search_window.fpp_min = 0.8 * best_guess[0];
    search_window.fpp_max = 1.2 * best_guess[0];
    search_window.gp_min = 0.8 * best_guess[1];
    search_window.gp_max = 1.2 * best_guess[1];

    // Update offset for edge conditions
    offset += 6;
  }
}