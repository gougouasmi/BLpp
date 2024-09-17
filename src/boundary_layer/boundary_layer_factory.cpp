#include "boundary_layer_factory.h"
#include "profile_functions_cpg.h"
#include "profile_functions_default.h"
#include "profile_struct.h"

BoundaryLayer BoundaryLayerFactory(const int grid_size,
                                   const std::string &type) {
  if (type == "default") {
    return BoundaryLayer(grid_size, default_model_functions);
  } else if (type == "cpg") {
    return BoundaryLayer(grid_size, cpg_model_functions);
  } else {
    printf("BoundaryLayerFactory: invalid type argument.");
    exit(0);
  }
}