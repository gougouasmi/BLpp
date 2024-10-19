#ifndef TIMERS_HPP
#define TIMERS_HPP

#include <chrono>
#include <ctime>

template <class T> double timeit(T &function, int nb_reps) {

  double avg_duration = 0.;
  for (int rep = 0; rep < nb_reps; rep++) {
    auto start = std::chrono::high_resolution_clock::now();

    function();

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    avg_duration += duration.count();
  }

  avg_duration *= 1. / nb_reps;

  return avg_duration;
}

template <class T> double timeit_clock(T &function, int nb_reps) {

  double avg_duration = 0.;
  for (int rep = 0; rep < nb_reps; rep++) {
    auto start = std::clock();

    function();

    auto end = std::clock();

    avg_duration += (end - start) / (double)CLOCKS_PER_SEC;
  }

  avg_duration *= 1. / nb_reps;

  return avg_duration;
}

#endif