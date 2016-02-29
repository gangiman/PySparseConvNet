#define RNG_CPP
#include "Rng.h"
#include <sys/time.h>
#include <time.h>

#include <mach/clock.h>
#include <mach/mach.h>

#include <iostream>
#include <algorithm>
#include <chrono>

RNG::RNG() : stdNormal(0, 1), uniform01(0, 1) {
  // auto seed =
  // std::chrono::high_resolution_clock::now().time_since_epoch().count();
  // std::mt19937 gen(seed);
  timespec ts;
#ifdef TARGET_OS_MAC
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts.tv_sec = mts.tv_sec;
  ts.tv_nsec = mts.tv_nsec;
#elif defined __linux__
  clock_gettime(CLOCK_REALTIME, &ts);
#endif

  RNGseedGeneratorMutex.lock();
  gen.seed(RNGseedGenerator() + ts.tv_nsec);
  RNGseedGeneratorMutex.unlock();
}

int RNG::randint(int n) {
  if (n == 0)
    return 0;
  else
    return gen() % n;
}
float RNG::uniform(float a, float b) { return a + (b - a) * uniform01(gen); }
float RNG::normal(float mean, float sd) { return mean + sd * stdNormal(gen); }
int RNG::bernoulli(float p) {
  if (uniform01(gen) < p)
    return 1;
  else
    return 0;
}
template <typename T> int RNG::index(std::vector<T> &v) {
  if (v.size() == 0)
    std::cout << "RNG::index called for empty std::vector!\n";
  return gen() % v.size();
}
std::vector<int> RNG::NchooseM(int n, int m) {
  std::vector<int> ret(m, 100);
  int ctr = m;
  for (int i = 0; i < n; i++)
    if (uniform01(gen) < ctr * 1.0 / (n - i))
      ret[m - ctr--] = i;
  return ret;
}
std::vector<int> RNG::permutation(int n) {
  std::vector<int> ret;
  for (int i = 0; i < n; i++)
    ret.push_back(i);
  std::shuffle(ret.begin(), ret.end(), gen);
  return ret;
}
template <typename T> void RNG::vectorShuffle(std::vector<T> &v) {
  std::shuffle(v.begin(), v.end(), gen);
}

template void RNG::vectorShuffle<int>(std::vector<int> &v);
