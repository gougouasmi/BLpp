#ifndef SCORE_FUNCTIONS_H
#define SCORE_FUNCTIONS_H

#include "profile_struct.hpp"
#include <vector>

using std::vector;

void ComputeScore(const ProfileParams &params, const vector<double> &state,
                  int state_offset, vector<double> &score);
void ComputeScoreJacobian(const ProfileParams &params,
                          const vector<double> &state, int state_offset,
                          const vector<double> &sensitivity_cm,
                          int sensitivity_offset,
                          vector<double> &score_jacobian_rm);

void ComputeScoreDefault(const ProfileParams &params,
                         const vector<double> &state, int state_offset,
                         vector<double> &score);
void ComputeScoreJacobianDefault(const ProfileParams &params,
                                 const vector<double> &state, int state_offset,
                                 const vector<double> &sensitivity_cm,
                                 int sensitivity_offset,
                                 vector<double> &score_jacobian_rm);

void ComputeScoreSquare(const ProfileParams &params,
                        const vector<double> &state, int state_offset,
                        vector<double> &score);
void ComputeScoreJacobianSquare(const ProfileParams &params,
                                const vector<double> &state, int state_offset,
                                const vector<double> &sensitivity_cm,
                                int sensitivity_offset,
                                vector<double> &score_jacobian_rm);

void ComputeScoreSquareSteady(const ProfileParams &params,
                              const vector<double> &state, int state_offset,
                              vector<double> &score);
void ComputeScoreJacobianSquareSteady(const ProfileParams &params,
                                      const vector<double> &state,
                                      int state_offset,
                                      const vector<double> &sensitivity_cm,
                                      int sensitivity_offset,
                                      vector<double> &score_jacobian_rm);

void ComputeScoreExp(const ProfileParams &params, const vector<double> &state,
                     int state_offset, vector<double> &score);
void ComputeScoreJacobianExp(const ProfileParams &params,
                             const vector<double> &state, int state_offset,
                             const vector<double> &sensitivity_cm,
                             int sensitivity_offset,
                             vector<double> &score_jacobian_rm);

void ComputeScoreExpScaled(const ProfileParams &params,
                           const vector<double> &state, int state_offset,
                           vector<double> &score);
void ComputeScoreJacobianExpScaled(const ProfileParams &params,
                                   const vector<double> &state,
                                   int state_offset,
                                   const vector<double> &sensitivity_cm,
                                   int sensitivity_offset,
                                   vector<double> &score_jacobian_rm);

#endif