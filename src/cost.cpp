#include "cost.hpp"
#include <cmath>

/*********************************
 *  cross entropy cost function  *
 *********************************/
double cross_entropy_f(const double p, const double q) {
  return -p * std::log(q) - (1 - p) * std::log(1 - q);
}

double cross_entropy_d1f(const double, const double q) {
  return -std::log(q) + std::log(1 - q);
}

double cross_entropy_d2f(const double p, const double q) {
  return -p / q + (1 - p) / (1 - q);
}
