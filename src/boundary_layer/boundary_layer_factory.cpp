#include "boundary_layer_factory.h"
#include "profile_functions_cpg.h"
#include "profile_struct.h"

BoundaryLayer BoundaryLayerFactory(const int grid_size,
                                   const std::string &type) {
  if (type == "default") {
    return BoundaryLayer(grid_size);
  } else if (type == "cpg") {
    return BoundaryLayer(grid_size, initialize_cpg, initialize_sensitivity_cpg,
                         compute_rhs_cpg, compute_lsim_rhs_cpg,
                         compute_full_rhs_cpg, compute_rhs_jacobian_cpg,
                         compute_lsim_rhs_jacobian_cpg,
                         compute_full_rhs_jacobian_cpg, limit_update_cpg);
  } else {
    printf("BoundaryLayerFactory: invalid type argument.");
    exit(0);
  }
}