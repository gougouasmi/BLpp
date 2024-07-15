#include "flat_plate.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <tuple>

FlatPlate::FlatPlate(int max_nb_steps)
    : _max_nb_steps(max_nb_steps), compute_rhs(compute_rhs_default),
      initialize(initialize_default) {

  state_grids.resize(_max_nb_workers);
  eta_grids.resize(_max_nb_workers);
  rhs_vecs.resize(_max_nb_workers);

  for (int worker_id = 0; worker_id < _max_nb_workers; worker_id++) {
    state_grids[worker_id].resize(FLAT_PLATE_RANK * (1 + _max_nb_steps));
    eta_grids[worker_id].resize((1 + _max_nb_steps));
    rhs_vecs[worker_id].resize(FLAT_PLATE_RANK);
  }
}

FlatPlate::FlatPlate(int max_nb_steps, RhsFunction compute_rhs_fun,
                     InitializeFunction init_fun)
    : _max_nb_steps(max_nb_steps), compute_rhs(compute_rhs_fun),
      initialize(init_fun) {

  state_grids.resize(_max_nb_workers);
  eta_grids.resize(_max_nb_workers);
  rhs_vecs.resize(_max_nb_workers);

  for (int worker_id = 0; worker_id < _max_nb_workers; worker_id++) {
    state_grids[worker_id].resize(FLAT_PLATE_RANK * (1 + _max_nb_steps));
    eta_grids[worker_id].resize((1 + _max_nb_steps));
    rhs_vecs[worker_id].resize(FLAT_PLATE_RANK);
  }
}

void FlatPlate::InitializeState(ProfileParams &profile_params, int worker_id) {
  initialize(profile_params, state_grids[0]);
}

int FlatPlate::DevelopProfile(ProfileParams &profile_params,
                              std::vector<double> &score, bool &converged,
                              int worker_id) {
  assert(profile_params.AreValid());

  vector<double> &state_grid = state_grids[worker_id];
  vector<double> &eta_grid = eta_grids[worker_id];
  vector<double> &rhs = rhs_vecs[worker_id];

  converged = false;
  int nb_steps = eta_grid.size() - 1;

  assert(state_grid.size() / (nb_steps + 1) == FLAT_PLATE_RANK);

  InitializeState(profile_params, worker_id);

  double min_step = profile_params.min_eta_step;
  double eta_step = 1.;

  int step_id = 0;
  int offset = 0;
  while (step_id < nb_steps) {

    // Compute rhs and limit time step
    eta_step = std::min(min_step,
                        compute_rhs(state_grid, rhs, offset, profile_params));

    // Evolve state/grid forward
    eta_grid[step_id + 1] = eta_grid[step_id] + eta_step;
    for (int var_id = 0; var_id < FLAT_PLATE_RANK; ++var_id) {
      state_grid[offset + FLAT_PLATE_RANK + var_id] =
          state_grid[offset + var_id] + eta_step * rhs[var_id];
    }

    // Update indexing
    offset += FLAT_PLATE_RANK;
    step_id += 1;

    // Check convergence
    double rate = sqrt(rhs[FP_ID] * rhs[FP_ID] + rhs[G_ID] * rhs[G_ID]);
    converged = rate < 1e-3;
    if (converged)
      break;
  }

  // Compute score
  score[0] = state_grid[offset + FP_ID] - 1.;
  score[1] = state_grid[offset + G_ID] - 1;

  return step_id;
}

void FlatPlate::BoxProfileSearch(ProfileParams &profile_params,
                                 SearchWindow &window,
                                 SearchParams &search_params,
                                 std::vector<double> &best_guess) {

  int max_iter = search_params.max_iter;
  double rtol = search_params.rtol;
  bool verbose = search_params.verbose;

  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  std::vector<double> initial_guess(2, 0.0);
  std::vector<double> score(2, 0.0);

  bool converged;

  double delta_fpp, delta_gp;

  double min_res_norm;
  double res_norm;

  int min_fid, min_gid;

  double fpp0, gp0;

  for (int iter = 0; iter < max_iter; iter++) {

    delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    delta_gp = (gp_max - gp_min) / (ydim - 1);

    fpp0 = fpp_min;
    gp0 = gp_min;

    min_res_norm = 1e3;

    min_fid = 0;
    min_gid = 0;

    // (1 / 2) Develop profiles on square grid of initial conditions
    for (int fid = 0; fid < xdim; fid++) {
      initial_guess[0] = fpp0;

      gp0 = gp_min;
      for (int gid = 0; gid < ydim; gid++) {
        initial_guess[1] = gp0;

        profile_params.SetInitialValues(initial_guess);
        InitializeState(profile_params);

        DevelopProfile(profile_params, score, converged);

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
              return;
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

    printf("Iter %d - score=%.2e, f''(0)=%.3f, g'(0)=%.3f, next window "
           "=[[%.3f, %.3f], [%.3f, %.3f]\n",
           iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
  }
}

#include <future>
#include <thread>

void FlatPlate::BoxProfileSearchParallel(ProfileParams &profile_params,
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

  bool converged;

  double delta_fpp, delta_gp;

  double min_res_norm;
  double res_norm;

  int min_fid, min_gid;

  for (int iter = 0; iter < max_iter; iter++) {

    delta_fpp = (fpp_max - fpp_min) / (xdim - 1);
    delta_gp = (gp_max - gp_min) / (ydim - 1);

    min_res_norm = 1e3;

    min_fid = 0;
    min_gid = 0;

    // (1 / 2) Develop profiles on square grid of initial conditions
    auto local_search_task =
        [fpp_min, gp_min, delta_fpp, delta_gp, rtol, profile_params,
         this](const int fid_start, const int fid_end, const int gid_start,
               const int gid_end, const int worker_id) mutable {
          bool converged = false;
          double fpp0 = fpp_min;
          double gp0 = gp_min;

          std::vector<double> initial_guess(2, 0.0);
          std::vector<double> score(2, 0.0);

          double res_norm;
          double min_res_norm = 1e30;

          int min_fid = 0, min_gid = 0;

          for (int fid = fid_start; fid < fid_end; fid++) {
            initial_guess[0] = fpp0;

            gp0 = gp_min;
            for (int gid = gid_start; gid < gid_end; gid++) {
              initial_guess[1] = gp0;

              profile_params.SetInitialValues(initial_guess);
              InitializeState(profile_params, worker_id);

              DevelopProfile(profile_params, score, converged, worker_id);

              if (converged) {
                res_norm = sqrt(score[0] * score[0] + score[1] * score[1]);

                if (res_norm < min_res_norm) {

                  min_res_norm = res_norm;
                  min_fid = fid;
                  min_gid = gid;

                  if (res_norm < rtol) {
                    return std::tuple<double, int, int>(min_res_norm, min_fid,
                                                        min_gid);
                  }
                }
              }

              gp0 += delta_gp;
            }

            fpp0 += delta_fpp;
          }
          return std::tuple<double, int, int>(min_res_norm, min_fid, min_gid);
        };

    // One thread involved (debugging)
    // auto ftr =
    // std::async(std::launch::async, local_search_task, 0, xdim, 0, ydim, 0);//
    // std::tuple < double, int, int> result = ftr.get();

    // 2 threads
    auto ftr1 = std::async(std::launch::async, local_search_task, 0,
                           (int)(xdim / 2), 0, ydim, 0);
    auto ftr2 = std::async(std::launch::async, local_search_task,
                           (int)(xdim / 2), xdim, 0, ydim, 1);

    using Result = std::tuple<double, int, int>;

    Result result_1 = ftr1.get();
    Result result_2 = ftr2.get();

    if (std::get<0>(result_1) < std::get<0>(result_2)) {
      min_res_norm = std::get<0>(result_1);
      min_fid = std::get<1>(result_1);
      min_gid = std::get<2>(result_1);
    } else {
      min_res_norm = std::get<0>(result_2);
      min_fid = std::get<1>(result_2);
      min_gid = std::get<2>(result_2);
    }

    if (min_res_norm < rtol) {
      best_guess[0] = fpp_min + min_fid * delta_fpp;
      best_guess[1] = gp_min + min_gid * delta_gp;
      printf("Solution found: f''(0)=%.6f, g'(0)=%.6f.\n", best_guess[0],
             best_guess[1]);

      return;
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

    printf("Iter %d - score=%.2e, f''(0)=%.3f, g'(0)=%.3f, next window "
           "=[[%.3f, %.3f], [%.3f, %.3f]\n",
           iter + 1, min_res_norm, xs, ys, fpp_min, fpp_max, gp_min, gp_max);
  }
}