#ifndef CASE_FUNCTIONS_H
#define CASE_FUNCTIONS_H

#include "boundary_data_struct.h"

BoundaryData GenFlatPlateConstant(double ue, double he, double pe, double g0,
                                  int nb_points);
BoundaryData GenChapmannRubesinFlatPlate(double mach, int nb_points,
                                         double prandtl = 0.72);
BoundaryData GetFlatNosedCylinder(double altitude, double mach,
                                  bool verbose = false);

#endif