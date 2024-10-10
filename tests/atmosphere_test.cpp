#include <cassert>
#include <cstdio>

#include "atmosphere.hpp"
#include "profile_struct.hpp"
#include "testing_utils.hpp"

void test_entry() {
  double rtol = 1e-4;

  ProfileParams profile_params;

  SetEntryConditions(17., 5., profile_params);

  assert(isClose(profile_params.pe, 8856.572616676596, rtol));
  assert(isClose(profile_params.ue, 5. * 300.13002682170935, rtol));
  assert(isClose(profile_params.he, 225195.08250000002, rtol));

  SetEntryConditions(42., 20., profile_params);

  assert(isClose(profile_params.pe, 224.69623222964393, rtol));
  assert(isClose(profile_params.ue, 20. * 333.47873095596367, rtol));
  assert(isClose(profile_params.he, 278020.16000000003, rtol));
}

int main() { test_entry(); }
