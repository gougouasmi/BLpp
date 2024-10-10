#include <cassert>
#include <cstdio>

#include "gas_model.hpp"
#include "testing_utils.hpp"

void test_thermo() {
  double rtol = 1e-4;

  assert(isClose(CP_AIR, 1039.25, rtol));
  assert(isClose(AIR_CPG_RO(1e6, 1e5), 0.35, rtol));

  assert(isClose(R_AIR, 296.92857142857144, rtol));

  double eps = 1e-2;
  assert(isClose(AIR_CPG_DRO_DH(1e6, 1e5),
                 (AIR_CPG_RO(1e6 + eps, 1e5) - AIR_CPG_RO(1e6, 1e5)) / eps,
                 rtol));
}

void test_transport() {
  double rtol = 1e-4;
  double eps = 1e-2;

  // Test air viscosity values
  assert(isClose(AIR_VISC(100), 6.9234423777739225e-06, rtol));
  assert(isClose(AIR_VISC(500), 2.673119251019732e-05, rtol));
  assert(isClose(AIR_VISC(1000), 4.158057381934453e-05, rtol));
  assert(isClose(AIR_VISC(5000), 0.00010105403585354255, rtol));
  assert(isClose(AIR_VISC(10000), 0.00014448089622295498, rtol));

  // Test air viscosity gradient values
  assert(AIR_VISC_GRAD(100) != 0);
  assert(AIR_VISC_GRAD(500) != 0);
  assert(AIR_VISC_GRAD(1000) != 0);
  assert(AIR_VISC_GRAD(5000) != 0);
  assert(AIR_VISC_GRAD(10000) != 0);

  assert(isClose(AIR_VISC_GRAD(100),
                 (AIR_VISC(100 + eps) - AIR_VISC(100)) / eps, rtol));
  assert(isClose(AIR_VISC_GRAD(500),
                 (AIR_VISC(500 + eps) - AIR_VISC(500)) / eps, rtol));
  assert(isClose(AIR_VISC_GRAD(1000),
                 (AIR_VISC(1000 + eps) - AIR_VISC(1000)) / eps, rtol));
  assert(isClose(AIR_VISC_GRAD(5000),
                 (AIR_VISC(5000 + eps) - AIR_VISC(5000)) / eps, rtol));
  assert(isClose(AIR_VISC_GRAD(10000),
                 (AIR_VISC(10000 + eps) - AIR_VISC(10000)) / eps, rtol));

  // Test air conductivity values
  assert(isClose(AIR_COND(100), 0.00848677370602224, rtol));
  assert(isClose(AIR_COND(500), 0.040196245370310676, rtol));
  assert(isClose(AIR_COND(1000), 0.06608237236057629, rtol));
  assert(isClose(AIR_COND(5000), 0.16984118597898148, rtol));
  assert(isClose(AIR_COND(10000), 0.2447627496145319, rtol));

  assert(AIR_COND_GRAD(100) != 0);
  assert(AIR_COND_GRAD(500) != 0);
  assert(AIR_COND_GRAD(1000) != 0);
  assert(AIR_COND_GRAD(5000) != 0);
  assert(AIR_COND_GRAD(10000) != 0);

  assert(isClose(AIR_COND_GRAD(100),
                 (AIR_COND(100 + eps) - AIR_COND(100)) / eps, rtol));
  assert(isClose(AIR_COND_GRAD(500),
                 (AIR_COND(500 + eps) - AIR_COND(500)) / eps, rtol));
  assert(isClose(AIR_COND_GRAD(1000),
                 (AIR_COND(1000 + eps) - AIR_COND(1000)) / eps, rtol));
  assert(isClose(AIR_COND_GRAD(5000),
                 (AIR_COND(5000 + eps) - AIR_COND(5000)) / eps, rtol));
  assert(isClose(AIR_COND_GRAD(10000),
                 (AIR_COND(10000 + eps) - AIR_COND(10000)) / eps, rtol));
}

int main() {
  test_thermo();
  test_transport();
}
