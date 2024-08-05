#ifndef ATMOSPHERE_H
#define ATMOSPHERE_H

#include "gas_model.h"
#include "profile_struct.h"
#include <cassert>
#include <cmath>

void inline SetEntryConditions(double altitude_km, double mach_number,
                               ProfileParams &profile_params) {
  assert(altitude_km > 0);

  double temperature, pressure;

  if (altitude_km > 25) { // Upper Stratosphere
    temperature = -131.21 + .00299 * (altitude_km * 1e3) + 273.15;
    pressure = 2.488e3 * pow(temperature / 216.6, -11.388);
  } else if (altitude_km > 11) { // Lower Stratosphere
    temperature = -56.46 + 273.15;
    pressure = 22.65e3 * exp(1.73 - 0.000157 * (altitude_km * 1e3));
  } else { // Troposphere
    temperature = 15.04 - 0.00649 * (altitude_km * 1e3) + 273.15;
    pressure = 101.29e3 * pow(temperature / 288.08, 5.256);
  }

  double speed_of_sound = sqrt(GAM * R_AIR * temperature);

  profile_params.pe = pressure;
  profile_params.ue = mach_number * speed_of_sound;
  profile_params.he = CP_AIR * temperature;
}

void ParseEntryParams(int argc, char *argv[], double &altitude_km,
                      double &mach_number) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-mach") {
      if (i + 1 < argc) {
        mach_number = std::stod(argv[++i]);
      } else {
        printf("mach number spec is incomplete\n");
      }
    } else if (arg == "-altitude") {
      if (i + 1 < argc) {
        altitude_km = std::stod(argv[++i]);
      } else {
        printf("altitude (km) spec is incomplete.\n");
      }
    }
  }
}

#endif