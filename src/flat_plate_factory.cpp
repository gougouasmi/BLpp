#include "flat_plate_factory.h"
#include "profile.h"

FlatPlate FlatPlateFactory(const int grid_size, const std::string &type) {
  if (type == "default") {
    return FlatPlate(grid_size);
  } else if (type == "cpg") {
    return FlatPlate(grid_size, compute_rhs_cpg, initialize_cpg);
  } else {
    printf("FlatPlateFactory: invalid type argument.");
    exit(0);
  }
}