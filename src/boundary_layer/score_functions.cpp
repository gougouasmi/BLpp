#include "score_functions.h"

void ComputeScore(Scoring method, const vector<double> &state, int state_offset,
                  vector<double> &score) {
  if (method == Scoring::Default) {
    ComputeScoreDefault(state, state_offset, score);
  }
}

void ComputeScoreJacobian(Scoring method, const vector<double> &state,
                          int state_offset,
                          const vector<double> &sensitivity_cm, int svty_offset,
                          vector<double> &score_jacobian_rm) {
  if (method == Scoring::Default) {
    ComputeScoreJacobianDefault(state, state_offset, sensitivity_cm,
                                svty_offset, score_jacobian_rm);
  }
}

void ComputeScoreDefault(const vector<double> &state, int state_offset,
                         vector<double> &score) {
  score[0] = state[state_offset + FP_ID] - 1.;
  score[1] = state[state_offset + G_ID] - 1;
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