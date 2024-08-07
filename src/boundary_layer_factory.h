#ifndef BOUNDARY_LAYER_FACTORY_H
#define BOUNDARY_LAYER_FACTORY_H

#include "boundary_layer.h"
#include <string>

BoundaryLayer BoundaryLayerFactory(const int grid_size,
                                   const std::string &type);

#endif