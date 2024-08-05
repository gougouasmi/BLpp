#include "flat_plate_factory.h"
#include "profile_functions_cpg.h"
#include "profile_struct.h"

FlatPlate FlatPlateFactory(const int grid_size, const std::string &type) {
  if (type == "default") {
    return FlatPlate(grid_size);
  } else if (type == "cpg") {
    return FlatPlate(grid_size, initialize_cpg, compute_rhs_cpg,
                     compute_lsim_rhs_cpg, compute_full_rhs_cpg,
                     compute_rhs_jacobian_cpg, compute_lsim_rhs_jacobian_cpg,
                     compute_full_rhs_jacobian_cpg);
  } else {
    printf("FlatPlateFactory: invalid type argument.");
    exit(0);
  }
}