#include <cassert>
#include <cmath>

#include "shock_relations.hpp"

void NormalShockRatios() {
  ShockRatios ratios = ComputeShockRatiosCPG(2.0);
  assert(ratios.pressure == 4.5);
  assert(ratios.velocity == 0.375);
  assert(ratios.density == 2.666666666666667);

  ratios = ComputeShockRatiosCPG(15.0);
  assert(ratios.pressure == 262.33333333333337);
  assert(ratios.velocity == 0.1703703703703704);
  assert(ratios.density == 5.8695652173913055);

  ratios = ComputeShockRatiosCPG(15.0, 1.1);
  assert(ratios.pressure == 235.66666666666669);
  assert(ratios.density == 19.28571428571427);
  assert(ratios.velocity == .051851851851851816);
}

void ObliqueShockRatios() {
  ShockRatios ratios = ComputeShockRatiosCPG(4.0, 1.7, 0.25 * M_PI);
  assert(ratios.pressure == 9.814814814814811);
  assert(ratios.density == 2.842105263157895);
  assert(ratios.velocity == 0.7137141305048574);
}

int main(int argc, char *argv[]) {
  NormalShockRatios();
  ObliqueShockRatios();
}