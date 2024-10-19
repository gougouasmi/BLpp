#ifndef BOUNDARY_LAYER_FACTORY_HPP
#define BOUNDARY_LAYER_FACTORY_HPP

#include "boundary_layer.hpp"
#include <string>

BoundaryLayer BoundaryLayerFactory(const int grid_size,
                                   const std::string &type);

#endif