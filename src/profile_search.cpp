#include <cassert>
#include <iostream>

#include "profile.h"
#include "profile_search.h"

/*
 *
 *
 *
 */
void box_profile_search(SearchWindow &window, SearchParams &params,
                        std::vector<double> &best_guess) {

  int max_iter = params.max_iter;
  double rtol = params.rtol;
  bool verbose = params.verbose;

  int xdim = window.xdim;
  int ydim = window.ydim;

  assert(xdim > 1);
  assert(ydim > 1);

  double fpp_min = window.fpp_min;
  double fpp_max = window.fpp_max;
  double gp_min = window.gp_min;
  double gp_max = window.gp_max;

  ProfileParams profile_params;
  profile_params.set_default();

  std::vector<double> initial_guess(2, 0.0);
  std::vector<double> rhs(5, 0.0);
  std::vector<double> score(2, 0.0);

  std::vector<double> state_grid(2000 * 5, 0.0);
  std::vector<double> eta_grid(2000, 0.0);

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

        profile_params.set_initial_values(initial_guess);

        develop_profile(profile_params, state_grid, eta_grid, rhs, score,
                        converged);

        res_norm = 1e3;
        if (converged)
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