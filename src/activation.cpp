#include <cmath>
#include "activation.hpp"

/*********************************
 *  sigmoid activation function  *
 *********************************/
double sigmoid_f(const double x)
{
    return 1 / (1 + std::exp(-x));
}
double sigmoid_df(const double x)
{
    return sigmoid_f(x) * (1 - sigmoid_f(x));
}

/******************************
 *  ReLu activation function  *
 ******************************/
double relu_f(const double x)
{
    return (x > 0) ? x : 0;
}
double relu_df(const double x)
{
    return (x > 0) ? 1 : 0;
}
