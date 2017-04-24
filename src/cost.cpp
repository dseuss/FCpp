#include <cmath>
#include "cost.hpp"

/*********************************
 *  cross entropy cost function  *
 *********************************/
double cross_entropy_f(const double p, const double q)
{
    return -p * std::log(q);
}

double cross_entropy_d1f(const double, const double q)
{
    return -std::log(q);
}

double cross_entropy_d2f(const double p, const double q)
{
    return -p / q;
}
