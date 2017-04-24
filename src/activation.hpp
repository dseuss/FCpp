#ifndef ACTIVATION_HPP_ZTOI6K93
#define ACTIVATION_HPP_ZTOI6K93

#include <cmath>

// TODO Can this be expressed as a class? There might be problems since Eigen
//      expects function pointers.
typedef struct ActivationFunction {
    double (*f)(const double);
    double (*df)(const double);
} ActivationFunction;


/*********************************
 *  sigmoid activation function  *
 *********************************/
double sigmoid_f(const double x)
{
    return 1 / (1 + std::exp(x));
}
double sigmoid_df(const double x)
{
    return sigmoid_f(x) * (1 - sigmoid_f(x));
}
const ActivationFunction sigmoid = { sigmoid_f, sigmoid_df };


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
const ActivationFunction relu = { relu_f, relu_df };


#endif /* end of include guard: ACTIVATION_HPP_ZTOI6K93 */
