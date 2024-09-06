#ifndef SCORE_FUNCTIONS_H
#define SCORE_FUNCTIONS_H

#include "profile_struct.h"
#include <vector>

using std::vector;

void ComputeScore(Scoring method, const vector<double> &state, int state_offset,
                  vector<double> &score);
void ComputeScoreJacobian(Scoring method, const vector<double> &state,
                          int state_offset,
                          const vector<double> &sensitivity_cm,
                          int sensitivity_offset,
                          vector<double> &score_jacobian_rm);

void ComputeScoreDefault(const vector<double> &state, int state_offset,
                         vector<double> &score);
void ComputeScoreJacobianDefault(const vector<double> &state, int state_offset,
                                 const vector<double> &sensitivity_cm,
                                 int sensitivity_offset,
                                 vector<double> &score_jacobian_rm);

#endif