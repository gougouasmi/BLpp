#include "score_functions.h"

void ComputeScore(Scoring method, const vector<double> &state, int state_offset,
                  vector<double> &score) {
  if (method == Scoring::Default) {
    ComputeScoreDefault(state, state_offset, score);
  } else if (method == Scoring::Square) {
    ComputeScoreSquare(state, state_offset, score);
  } else if (method == Scoring::SquareSteady) {
    ComputeScoreSquareSteady(state, state_offset, score);
  } else if (method == Scoring::Exp) {
    ComputeScoreExp(state, state_offset, score);
  }
}

void ComputeScoreJacobian(Scoring method, const vector<double> &state,
                          int state_offset,
                          const vector<double> &sensitivity_cm, int svty_offset,
                          vector<double> &score_jacobian_rm) {
  if (method == Scoring::Default) {
    ComputeScoreJacobianDefault(state, state_offset, sensitivity_cm,
                                svty_offset, score_jacobian_rm);
  } else if (method == Scoring::Square) {
    ComputeScoreJacobianSquare(state, state_offset, sensitivity_cm, svty_offset,
                               score_jacobian_rm);
  } else if (method == Scoring::SquareSteady) {
    ComputeScoreJacobianSquareSteady(state, state_offset, sensitivity_cm,
                                     svty_offset, score_jacobian_rm);
  } else if (method == Scoring::Exp) {
    ComputeScoreJacobianExp(state, state_offset, sensitivity_cm, svty_offset,
                            score_jacobian_rm);
  }
}

//

void ComputeScoreDefault(const vector<double> &state, int state_offset,
                         vector<double> &score) {
  score[0] = state[state_offset + FP_ID] - 1.;
  score[1] = state[state_offset + G_ID] - 1.;
}

void ComputeScoreJacobianDefault(const vector<double> &state, int state_offset,
                                 const vector<double> &sensitivity_cm,
                                 int svty_offset,
                                 vector<double> &score_jacobian_rm) {
  score_jacobian_rm[0] = sensitivity_cm[svty_offset + 0 + FP_ID];
  score_jacobian_rm[1] = sensitivity_cm[svty_offset + BL_RANK + FP_ID];

  score_jacobian_rm[2 + 0] = sensitivity_cm[svty_offset + 0 + G_ID];
  score_jacobian_rm[2 + 1] = sensitivity_cm[svty_offset + BL_RANK + G_ID];
}

//

void ComputeScoreSquare(const vector<double> &state, int state_offset,
                        vector<double> &score) {
  score[0] =
      (state[state_offset + FP_ID] - 1.) * (state[state_offset + FP_ID] - 1.);
  score[1] =
      (state[state_offset + G_ID] - 1.) * (state[state_offset + G_ID] - 1.);
}

void ComputeScoreJacobianSquare(const vector<double> &state, int state_offset,
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

void ComputeScoreSquareSteady(const vector<double> &state, int state_offset,
                              vector<double> &score) {

  double fp_error = state[state_offset + FP_ID] - 1.;
  double g_error = state[state_offset + G_ID] - 1.;

  double fpp_error = state[state_offset + FPP_ID];
  double gp_error = state[state_offset + GP_ID];

  score[0] = fp_error * fp_error + fpp_error * fpp_error;
  score[1] = g_error * g_error + gp_error * gp_error;
}

void ComputeScoreJacobianSquareSteady(const vector<double> &state,
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

//

void ComputeScoreExp(const vector<double> &state, int state_offset,
                     vector<double> &score) {

  double fp_error = state[state_offset + FP_ID] - 1.;
  double g_error = state[state_offset + G_ID] - 1.;

  double fpp_error = state[state_offset + FPP_ID];
  double gp_error = state[state_offset + GP_ID];

  score[0] = fp_error + fpp_error;
  score[1] = g_error + gp_error;
}

void ComputeScoreJacobianExp(const vector<double> &state, int state_offset,
                             const vector<double> &sensitivity_cm,
                             int svty_offset,
                             vector<double> &score_jacobian_rm) {

  double fp_error = state[state_offset + FP_ID] - 1.;
  double g_error = state[state_offset + G_ID] - 1.;

  double fpp_error = state[state_offset + FPP_ID];
  double gp_error = state[state_offset + GP_ID];

  score_jacobian_rm[0] = 2. * (0.5 * sensitivity_cm[svty_offset + 0 + FP_ID] +
                               0.5 * sensitivity_cm[svty_offset + 0 + FPP_ID]);

  score_jacobian_rm[1] =
      2. * (0.5 * sensitivity_cm[svty_offset + BL_RANK + FP_ID] +
            0.5 * sensitivity_cm[svty_offset + BL_RANK + FPP_ID]);

  score_jacobian_rm[2 + 0] =
      2. * (0.5 * sensitivity_cm[svty_offset + 0 + G_ID] +
            0.5 * sensitivity_cm[svty_offset + 0 + GP_ID]);

  score_jacobian_rm[2 + 1] =
      2. * (0.5 * sensitivity_cm[svty_offset + BL_RANK + G_ID] +
            0.5 * sensitivity_cm[svty_offset + BL_RANK + GP_ID]);
}