#ifndef FLAT_PLATE_FACTORY_H
#define FLAT_PLATE_FACTORY_H

#include "flat_plate.h"
#include <string>

FlatPlate FlatPlateFactory(const int grid_size, const std::string &type);

#endif