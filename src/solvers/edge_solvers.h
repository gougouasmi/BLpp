#ifndef EDGE_SOLVER_H
#define EDGE_SOLVER_H

#include <vector>

using std::vector;

void ComputeFromPressure(const vector<double> &pressure_field,
                         vector<double> &density_field,
                         vector<double> &velocity_field);

#endif