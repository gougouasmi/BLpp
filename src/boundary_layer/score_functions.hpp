#ifndef SCORE_FUNCTIONS_H
#define SCORE_FUNCTIONS_H

#include "profile_struct.hpp"
#include <vector>

using std::vector;

void ComputeScore(ProfileParams &params, const vector<double> &state,
                  int state_offset, vector<double> &score);
void ComputeScoreJacobian(ProfileParams &params, const vector<double> &state,
                          int state_offset,
                          const vector<double> &sensitivity_cm,
                          int sensitivity_offset,
                          vector<double> &score_jacobian_rm);

void ComputeScoreDefault(ProfileParams &params, const vector<double> &state,
                         int state_offset, vector<double> &score);
void ComputeScoreJacobianDefault(ProfileParams &params,
                                 const vector<double> &state, int state_offset,
                                 const vector<double> &sensitivity_cm,
                                 int sensitivity_offset,
                                 vector<double> &score_jacobian_rm);

void ComputeScoreSquare(ProfileParams &params, const vector<double> &state,
                        int state_offset, vector<double> &score);
void ComputeScoreJacobianSquare(ProfileParams &params,
                                const vector<double> &state, int state_offset,
                                const vector<double> &sensitivity_cm,
                                int sensitivity_offset,
                                vector<double> &score_jacobian_rm);

void ComputeScoreSquareSteady(ProfileParams &params,
                              const vector<double> &state, int state_offset,
                              vector<double> &score);
void ComputeScoreJacobianSquareSteady(ProfileParams &params,
                                      const vector<double> &state,
                                      int state_offset,
                                      const vector<double> &sensitivity_cm,
                                      int sensitivity_offset,
                                      vector<double> &score_jacobian_rm);

void ComputeScoreExp(ProfileParams &params, const vector<double> &state,
                     int state_offset, vector<double> &score);
void ComputeScoreJacobianExp(ProfileParams &params, const vector<double> &state,
                             int state_offset,
                             const vector<double> &sensitivity_cm,
                             int sensitivity_offset,
                             vector<double> &score_jacobian_rm);

void ComputeScoreExpScaled(ProfileParams &params, const vector<double> &state,
                           int state_offset, vector<double> &score);
void ComputeScoreJacobianExpScaled(ProfileParams &params,
                                   const vector<double> &state,
                                   int state_offset,
                                   const vector<double> &sensitivity_cm,
                                   int sensitivity_offset,
                                   vector<double> &score_jacobian_rm);

#endif