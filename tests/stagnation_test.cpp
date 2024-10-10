#include <cassert>
#include <cmath>

#include "stagnation.hpp"

void StagnationRatios() {
  double mach, gamma;
  double temperature_ratio, pressure_ratio, density_ratio;

  mach = 2.;
  gamma = 1.4;
  ComputeStagnationRatios(mach, gamma, temperature_ratio, density_ratio,
                          pressure_ratio);

  assert(temperature_ratio == 1.7999999999999998);
  assert(pressure_ratio == 7.824449066867263);
  assert(pressure_ratio * pow(density_ratio, -gamma) == 1);

  mach = 8.;
  gamma = 1.19;

  ComputeStagnationRatios(mach, gamma, temperature_ratio, density_ratio,
                          pressure_ratio);

  assert(temperature_ratio == 7.079999999999998);
  assert(pressure_ratio == 210810.63124083486);
  assert(pressure_ratio * pow(density_ratio, -gamma) == 1);
}

int main(int argc, char *argv[]) { StagnationRatios(); }