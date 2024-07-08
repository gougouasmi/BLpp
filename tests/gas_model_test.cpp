#include <cassert>
#include <cstdio>

#include "gas_model.h"
#include "testing_utils.h"

void test_thermo() {
  double rtol = 1e-4;

  assert(isClose(CP_AIR, 1039.25, rtol));
  assert(isClose(AIR_CPG_RO(1e6, 1e5), 0.35, rtol));

  assert(isClose(R_AIR, 296.92857142857144, rtol));
}

void test_transport() {
  double rtol = 1e-4;

  // Test air viscosity values
  assert(isClose(AIR_VISC(100), 6.9234423777739225e-06, rtol));
  assert(isClose(AIR_VISC(500), 2.673119251019732e-05, rtol));
  assert(isClose(AIR_VISC(1000), 4.158057381934453e-05, rtol));
  assert(isClose(AIR_VISC(5000), 0.00010105403585354255, rtol));
  assert(isClose(AIR_VISC(10000), 0.00014448089622295498, rtol));

  // Test air conductivity values
  assert(isClose(AIR_COND(100), 0.00848677370602224, rtol));
  assert(isClose(AIR_COND(500), 0.040196245370310676, rtol));
  assert(isClose(AIR_COND(1000), 0.06608237236057629, rtol));
  assert(isClose(AIR_COND(5000), 0.16984118597898148, rtol));
  assert(isClose(AIR_COND(10000), 0.2447627496145319, rtol));
}

int main() {
  test_thermo();
  test_transport();
}
