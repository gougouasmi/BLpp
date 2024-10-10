#include "score_functions.hpp"
#include "gas_model.hpp"

#define COMPUTE_SCORE_CASE(method)                                             \
  case Scoring::method:                                                        \
    ComputeScore##method(profile_params, state, state_offset, score);          \
    break;

#define COMPUTE_SCORE_JACOBIAN_CASE(method)                                    \
  case Scoring::method:                                                        \
    ComputeScoreJacobian##method(profile_params, state, state_offset,          \
                                 sensitivity_cm, svty_offset,                  \
                                 score_jacobian_rm);                           \
    break;

void ComputeScore(ProfileParams &profile_params, const vector<double> &state,
                  int state_offset, vector<double> &score) {
  switch (profile_params.scoring) {
    COMPUTE_SCORE_CASE(Default)
    COMPUTE_SCORE_CASE(Square)
    COMPUTE_SCORE_CASE(SquareSteady)
    COMPUTE_SCORE_CASE(Exp)
    COMPUTE_SCORE_CASE(ExpScaled)
  default:
    break;
  }
}

void ComputeScoreJacobian(ProfileParams &profile_params,
                          const vector<double> &state, int state_offset,
                          const vector<double> &sensitivity_cm, int svty_offset,
                          vector<double> &score_jacobian_rm) {
  switch (profile_params.scoring) {
    COMPUTE_SCORE_JACOBIAN_CASE(Default)
    COMPUTE_SCORE_JACOBIAN_CASE(Square)
    COMPUTE_SCORE_JACOBIAN_CASE(SquareSteady)
    COMPUTE_SCORE_JACOBIAN_CASE(Exp)
    COMPUTE_SCORE_JACOBIAN_CASE(ExpScaled)
  default:
    break;
  }
}

//

void ComputeScoreDefault(ProfileParams &profile_params,
                         const vector<double> &state, int state_offset,
                         vector<double> &score) {
  score[0] = state[state_offset + FP_ID] - 1.;
  score[1] = state[state_offset + G_ID] - 1.;
}

void ComputeScoreJacobianDefault(ProfileParams &profile_params,
                                 const vector<double> &state, int state_offset,
                                 const vector<double> &sensitivity_cm,
                                 int svty_offset,
                                 vector<double> &score_jacobian_rm) {
  score_jacobian_rm[0] = sensitivity_cm[svty_offset + 0 + FP_ID];
  score_jacobian_rm[1] = sensitivity_cm[svty_offset + BL_RANK + FP_ID];

  score_jacobian_rm[2 + 0] = sensitivity_cm[svty_offset + 0 + G_ID];
  score_jacobian_rm[2 + 1] = sensitivity_cm[svty_offset + BL_RANK + G_ID];
}

//

void ComputeScoreSquare(ProfileParams &profile_params,
                        const vector<double> &state, int state_offset,
                        vector<double> &score) {
  score[0] =
      (state[state_offset + FP_ID] - 1.) * (state[state_offset + FP_ID] - 1.);
  score[1] =
      (state[state_offset + G_ID] - 1.) * (state[state_offset + G_ID] - 1.);
}

void ComputeScoreJacobianSquare(ProfileParams &profile_params,
                                const vector<double> &state, int state_offset,
                                const vector<double> &sensitivity_cm,
                                int svty_offset,
                                vector<double> &score_jacobian_rm) {
  score_jacobian_rm[0] = 2. * (state[state_offset + FP_ID] - 1.) *
                         sensitivity_cm[svty_offset + 0 + FP_ID];
  score_jacobian_rm[1] = 2. * (state[state_offset + FP_ID] - 1.) *
                         sensitivity_cm[svty_offset + BL_RANK + FP_ID];

  score_jacobian_rm[2 + 0] = 2. * (state[state_offset + G_ID] - 1.) *
                             sensitivity_cm[svty_offset + 0 + G_ID];
  score_jacobian_rm[2 + 1] = 2. * (state[state_offset + G_ID] - 1.) *
                             sensitivity_cm[svty_offset + BL_RANK + G_ID];
}

//

void ComputeScoreSquareSteady(ProfileParams &profile_params,
                              const vector<double> &state, int state_offset,
                              vector<double> &score) {

  double fp_error = state[state_offset + FP_ID] - 1.;
  double g_error = state[state_offset + G_ID] - 1.;

  double fpp_error = state[state_offset + FPP_ID];
  double gp_error = state[state_offset + GP_ID];

  score[0] = fp_error * fp_error + fpp_error * fpp_error;
  score[1] = g_error * g_error + gp_error * gp_error;
}

void ComputeScoreJacobianSquareSteady(ProfileParams &profile_params,
                                      const vector<double> &state,
                                      int state_offset,
                                      const vector<double> &sensitivity_cm,
                                      int svty_offset,
                                      vector<double> &score_jacobian_rm) {

  double fp_error = state[state_offset + FP_ID] - 1.;
  double g_error = state[state_offset + G_ID] - 1.;

  double fpp_error = state[state_offset + FPP_ID];
  double gp_error = state[state_offset + GP_ID];

  score_jacobian_rm[0] =
      2. * (fp_error * sensitivity_cm[svty_offset + 0 + FP_ID] +
            fpp_error * sensitivity_cm[svty_offset + 0 + FPP_ID]);

  score_jacobian_rm[1] =
      2. * (fp_error * sensitivity_cm[svty_offset + BL_RANK + FP_ID] +
            fpp_error * sensitivity_cm[svty_offset + BL_RANK + FPP_ID]);

  score_jacobian_rm[2 + 0] =
      2. * (g_error * sensitivity_cm[svty_offset + 0 + G_ID] +
            gp_error * sensitivity_cm[svty_offset + 0 + GP_ID]);

  score_jacobian_rm[2 + 1] =
      2. * (g_error * sensitivity_cm[svty_offset + BL_RANK + G_ID] +
            gp_error * sensitivity_cm[svty_offset + BL_RANK + GP_ID]);
}

void ComputeScoreExp(ProfileParams &profile_params, const vector<double> &state,
                     int state_offset, vector<double> &score) {
  double g = state[state_offset + G_ID];
  double fp = state[state_offset + FP_ID];

  double Cfpp = state[state_offset + FPP_ID];
  double Cgp = state[state_offset + GP_ID];

  //
  double fp_error = fp - 1.;
  double g_error = g - 1.;

  double fpp_error = Cfpp;
  double gp_error = Cgp;

  score[0] = fp_error + fpp_error;
  score[1] = g_error + gp_error;
}

void ComputeScoreJacobianExp(ProfileParams &profile_params,
                             const vector<double> &state, int state_offset,
                             const vector<double> &sensitivity_cm,
                             int svty_offset,
                             vector<double> &score_jacobian_rm) {
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double Cfpp = state[state_offset + FPP_ID];
  double Cgp = state[state_offset + GP_ID];

  //
  double fp_error = fp - 1.;
  double g_error = g - 1.;

  double fpp_error = Cfpp;
  double gp_error = Cgp;

  // score[0] = fp_error + fpp_error;
  //   gradient wrt to f''(0)
  score_jacobian_rm[0] = (sensitivity_cm[svty_offset + 0 + FP_ID] +
                          sensitivity_cm[svty_offset + 0 + FPP_ID]);
  //   graident wrt to g'(0) or g(0)
  score_jacobian_rm[1] = (sensitivity_cm[svty_offset + BL_RANK + FP_ID] +
                          sensitivity_cm[svty_offset + BL_RANK + FPP_ID]);

  // score[1] = g_error + gp_error;
  //   gradient wrt to f''(0)
  score_jacobian_rm[2 + 0] = (sensitivity_cm[svty_offset + 0 + G_ID] +
                              sensitivity_cm[svty_offset + 0 + GP_ID]);

  //   gradient wrt to g'(0) or g(0)
  score_jacobian_rm[2 + 1] = (sensitivity_cm[svty_offset + BL_RANK + G_ID] +
                              sensitivity_cm[svty_offset + BL_RANK + GP_ID]);
}

//

void ComputeScoreExpScaled(ProfileParams &profile_params,
                           const vector<double> &state, int state_offset,
                           vector<double> &score) {
  double g = state[state_offset + G_ID];
  double fp = state[state_offset + FP_ID];

  double pe = profile_params.pe;
  double he = profile_params.he;
  double roe = profile_params.roe;
  double mue = profile_params.mue;

  // ro, temperature, cp = thermo_fun(pe, he * g)
  // mu, k = transport_fun(temperature)
  double ro = AIR_CPG_RO(g * he, pe);
  double cp = AIR_CPG_CP(g * he, pe);
  double temperature = pe / (ro * R_AIR);

  double mu = AIR_VISC(temperature);
  double k = AIR_COND(temperature);

  double romu = (ro * mu) / (roe * mue);
  double prandtl = mu * cp / k;

  double fpp = state[state_offset + FPP_ID] / romu;
  double gp = state[state_offset + GP_ID] / romu * prandtl;

  //
  double fp_error = fp - 1.;
  double g_error = g - 1.;

  double fpp_error = fpp;
  double gp_error = gp;

  score[0] = fp_error + fpp_error;
  score[1] = g_error + gp_error;
}

void ComputeScoreJacobianExpScaled(ProfileParams &profile_params,
                                   const vector<double> &state,
                                   int state_offset,
                                   const vector<double> &sensitivity_cm,
                                   int svty_offset,
                                   vector<double> &score_jacobian_rm) {
  double fp = state[state_offset + FP_ID];
  double g = state[state_offset + G_ID];

  double pe = profile_params.pe;
  double he = profile_params.he;
  double roe = profile_params.roe;
  double mue = profile_params.mue;

  // ro, temperature, cp = thermo_fun(pe, he * g)
  // mu, k = transport_fun(temperature)
  double ro = AIR_CPG_RO(g * he, pe);
  double dro_dg = AIR_CPG_DRO_DH(g * he, pe) * he;

  double cp = AIR_CPG_CP(g * he, pe);

  double temperature = pe / (ro * R_AIR);
  double dtemp_dg = -pe * dro_dg / (ro * ro * R_AIR);

  double mu = AIR_VISC(temperature);
  double k = AIR_COND(temperature);

  double dmu_dg = AIR_VISC_GRAD(temperature) * dtemp_dg;
  double dk_dg = AIR_COND_GRAD(temperature) * dtemp_dg;

  double romu = (ro * mu) / (roe * mue);
  double prandtl = mu * cp / k;

  double dromu_dg = (dro_dg * mu + ro * dmu_dg) / (roe * mue);
  double dprandtl_dg = (dmu_dg / k - mu * dk_dg / (k * k)) * cp;

  double fpp = state[state_offset + FPP_ID] / romu;
  double dfpp_dfpp = 1. / romu;
  double dfpp_dg = -dromu_dg * fpp / romu;

  double gp = state[state_offset + GP_ID] / romu * prandtl;
  double dgp_dgp = prandtl / romu;
  double dgp_dg = state[state_offset + GP_ID] *
                  (dprandtl_dg / romu - dromu_dg * prandtl / (romu * romu));

  //
  double fp_error = fp - 1.;
  double g_error = g - 1.;

  double fpp_error = fpp;
  double gp_error = gp;

  // score[0] = fp_error + fpp_error;
  //   gradient wrt to f''(0)
  score_jacobian_rm[0] = (1. * sensitivity_cm[svty_offset + 0 + FP_ID] +
                          dfpp_dfpp * sensitivity_cm[svty_offset + 0 + FPP_ID] +
                          dfpp_dg * sensitivity_cm[svty_offset + 0 + G_ID]);

  //   graident wrt to g'(0) or g(0)
  score_jacobian_rm[1] =
      (1.0 * sensitivity_cm[svty_offset + BL_RANK + FP_ID] +
       dfpp_dfpp * sensitivity_cm[svty_offset + BL_RANK + FPP_ID] +
       dfpp_dg * sensitivity_cm[svty_offset + BL_RANK + G_ID]);

  // score[1] = g_error + gp_error;
  //   gradient wrt to f''(0)
  score_jacobian_rm[2 + 0] =
      (1. * sensitivity_cm[svty_offset + 0 + G_ID] +
       dgp_dgp * sensitivity_cm[svty_offset + 0 + GP_ID] +
       dgp_dg * sensitivity_cm[svty_offset + 0 + G_ID]);

  //   gradient wrt to g'(0) or g(0)
  score_jacobian_rm[2 + 1] =
      (1. * sensitivity_cm[svty_offset + BL_RANK + G_ID] +
       dgp_dgp * sensitivity_cm[svty_offset + BL_RANK + GP_ID] +
       dgp_dg * sensitivity_cm[svty_offset + BL_RANK + G_ID]);
}