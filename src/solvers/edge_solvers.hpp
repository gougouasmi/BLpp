#ifndef EDGE_SOLVER_H
#define EDGE_SOLVER_H

#include <vector>

using std::vector;

int ComputeFromPressureBE(const vector<double> &pressure_field,
                          vector<double> &density_field,
                          vector<double> &velocity_field,
                          double gamma_ref = 1.4);

int ComputeFromPressureCN(const vector<double> &pressure_field,
                          vector<double> &density_field,
                          vector<double> &velocity_field,
                          double gamma_ref = 1.4);

int ComputeFromPressureConstantDensity(const vector<double> &pressure_field,
                                       vector<double> &density_field,
                                       vector<double> &velocity_field,
                                       double gamma_ref = 1.4);
#endif